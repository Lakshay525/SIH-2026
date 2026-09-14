"""
Parses the real Mozilla Public Suffix List (vendored at
data/public_suffix_list.dat, MPL-2.0, https://publicsuffix.org/list/) offline
and answers "what is the registrable label of this domain?".

Replaces the hand-maintained `_TWO_LABEL_SUFFIXES`/`_PRIVATE_SUFFIXES`/
`_GENERIC_SLDS` sets that used to live in lexical.py. Those were built by
re-deriving a subset of exactly what the real PSL already contains (including
"co.in", "co.uk", and every dynamic-DNS provider actually present in
data/dga_dataset.parquet, all of which are natively in the PSL's PRIVATE
section) -- using the real list is strictly more complete and removes an
entire category of manual-maintenance drift.

Deliberately a vendored SNAPSHOT FILE, not `tldextract`: tldextract fetches
this same list over the network on first use, which would violate the
read-only/no-egress constraint this whole system rests on
(tests/test_one_way_constraints.py fails it). Parsing a file already on disk
is offline by construction. The snapshot needs periodic manual refresh
(data/public_suffix_list.dat carries its own VERSION header) -- a real
deployment would re-vendor it on a release cadence, not auto-update live.

Implements the actual PSL algorithm (https://publicsuffix.org/list/), not an
approximation:
  1. The public suffix is the set of labels matching the longest rule.
  2. "*.example" matches any single prepended label unless overridden.
  3. "!exception.example" removes exactly that name from the wildcard match.
  4. No matching rule -> the last label alone is the public suffix.
  5. The registrable domain is the public suffix plus one label to its left.
"""
from pathlib import Path

_WILDCARD_PREFIX = "*."
_EXCEPTION_PREFIX = "!"


def _to_ascii(rule_text: str) -> str:
    """
    The PSL file stores IDN suffixes in Unicode (e.g. "公司.cn"), but a real
    DNS query name arrives in punycode/A-label form on the wire
    ("xn--55qx5d.cn") -- Python's builtin 'idna' codec does the same
    per-label ToASCII conversion real resolvers use. "*" is preserved as-is
    since it isn't a real label. Falls back to the original text for the
    (rare) label the idna codec rejects outright, rather than dropping it.
    """
    out_labels = []
    for label in rule_text.split("."):
        if label == "*" or label.isascii():
            out_labels.append(label.lower())
        else:
            try:
                out_labels.append(label.encode("idna").decode("ascii").lower())
            except UnicodeError:
                out_labels.append(label.lower())
    return ".".join(out_labels)


def parse_psl(path: Path):
    """
    Returns (rules, exceptions): rules is a set of suffix strings in PSL
    rule form (dots joined, "*" standing for one wildcard label, e.g.
    "*.ck"); exceptions is a set of full exception domains (e.g. "www.ck").
    Both are normalized to ASCII/punycode form -- see _to_ascii().
    """
    rules, exceptions = set(), set()
    text = Path(path).read_text(encoding="utf-8")
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("//"):
            continue
        # Real DNS query names arrive lowercased/punycode; normalize rules to
        # match (see registrable_domain(), which lowercases its input).
        if line.startswith(_EXCEPTION_PREFIX):
            exceptions.add(_to_ascii(line[len(_EXCEPTION_PREFIX):]))
        else:
            rules.add(_to_ascii(line))
    return rules, exceptions


class PublicSuffixList:
    def __init__(self, rules, exceptions):
        self._rules = rules
        self._exceptions = exceptions

    def __len__(self):
        return len(self._rules)

    @classmethod
    def from_file(cls, path: Path, extra_rules=()):
        rules, exceptions = parse_psl(path)
        rules |= {_to_ascii(r) for r in extra_rules}
        return cls(rules, exceptions)

    def public_suffix_length(self, labels: list) -> int:
        """
        How many trailing labels of `labels` make up the public suffix, per
        the PSL algorithm. Always >= 1: a name with no matching rule falls
        back to its bare last label (rule 4 above). Labels are normalized to
        ASCII/punycode here (matching how the rules themselves were parsed --
        see _to_ascii()) so a rare real Unicode domain and the (far more
        common) already-punycode wire form both look rules up consistently.
        """
        labels = [_to_ascii(l) for l in labels]
        best = 1  # implicit "*" rule: the last label alone

        for n in range(1, len(labels) + 1):
            candidate_labels = labels[-n:]
            candidate = ".".join(candidate_labels)

            if candidate in self._exceptions:
                # An exception rule is itself one label SHORTER than the
                # wildcard it overrides (e.g. "!city.kawasaki.jp" means
                # "city.kawasaki.jp" is NOT under the "*.kawasaki.jp" suffix
                # -- "kawasaki.jp" is the suffix instead).
                best = max(best, n - 1)
                continue

            if candidate in self._rules:
                best = max(best, n)
                continue

            if n >= 2:
                wildcard = "*." + ".".join(candidate_labels[1:])
                if wildcard in self._rules:
                    best = max(best, n)

        return best

    def registrable_domain(self, domain: str) -> str:
        """The public suffix plus exactly one label to its left."""
        labels = domain.rstrip(".").lower().split(".")
        if len(labels) < 2:
            return labels[0]
        suffix_len = min(self.public_suffix_length(labels), len(labels) - 1)
        return ".".join(labels[-(suffix_len + 1):])

    def registrable_label(self, domain: str) -> str:
        """Just the leftmost label of the registrable domain."""
        labels = domain.rstrip(".").lower().split(".")
        if len(labels) < 2:
            return labels[0]
        suffix_len = min(self.public_suffix_length(labels), len(labels) - 1)
        return labels[-(suffix_len + 1)]
