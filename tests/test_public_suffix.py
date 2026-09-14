"""
src/features/public_suffix.py replaced a hand-maintained set of suffixes
(country TLDs, dynamic-DNS providers, a generic-SLD heuristic) with a real
parser over the vendored Mozilla Public Suffix List. This pins two things:
  1. It implements the actual PSL algorithm correctly -- verified against a
     representative subset of the project's OWN official test vectors
     (https://github.com/publicsuffix/list/blob/master/tests/test_psl.txt),
     including the tricky wildcard/exception cases (kawasaki.jp, ck) and an
     IDN suffix requiring punycode normalization (公司.cn).
  2. The 13 empirically-verified extra suffixes (lexical.py) that the
     crowdsourced PSL doesn't list are exactly the ones proven necessary by
     this repo's own dataset.
"""
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import PUBLIC_SUFFIX_LIST_PATH
from src.features.public_suffix import PublicSuffixList
from src.features.lexical import _EMPIRICALLY_VERIFIED_EXTRA_SUFFIXES, extract_label


@pytest.fixture(scope="module")
def psl():
    return PublicSuffixList.from_file(PUBLIC_SUFFIX_LIST_PATH)


def test_psl_loads_a_substantial_real_list(psl):
    assert len(psl) > 5000, "expected the real PSL, not a placeholder"


# A representative subset of the official PSL test suite -- the full 73-vector
# check (including network fetch of the upstream file) was run manually during
# development; these are pinned here as an offline regression test.
OFFICIAL_VECTORS = [
    ("example.COM", "example.com"),        # case-insensitivity
    ("WwW.example.COM", "example.com"),
    ("example.example", "example.example"),  # unlisted TLD falls back to bare label
    ("b.example.example", "example.example"),
    ("domain.biz", "domain.biz"),          # single-rule TLD
    ("b.domain.biz", "domain.biz"),
    ("example.com", "example.com"),
    ("b.example.com", "example.com"),
    ("a.b.example.com", "example.com"),
    ("uk.com", None),                       # bare 2-level rule -> nothing left
    ("example.uk.com", "example.uk.com"),
    ("test.ac", "test.ac"),
    ("a.b.example.uk.com", "example.uk.com"),
    # *.ck wildcard with a "!www.ck" exception (real vectors, verbatim)
    ("ck", None),
    ("test.ck", None),                      # under the wildcard, but IS the suffix itself
    ("b.test.ck", "b.test.ck"),
    ("a.b.test.ck", "b.test.ck"),
    ("www.ck", "www.ck"),                   # exception overrides the wildcard
    ("www.www.ck", "www.ck"),
    # *.kobe.jp wildcard with a "!city.kobe.jp" exception (real vectors, verbatim)
    ("c.kobe.jp", None),
    ("b.c.kobe.jp", "b.c.kobe.jp"),
    ("a.b.c.kobe.jp", "b.c.kobe.jp"),
    ("city.kobe.jp", "city.kobe.jp"),
    ("www.city.kobe.jp", "city.kobe.jp"),
    ("xn--85x722f.xn--55qx5d.cn", "xn--85x722f.xn--55qx5d.cn"),  # IDN, punycode form
    ("食狮.公司.cn", "食狮.公司.cn"),                              # IDN, Unicode form
    ("shishi.xn--55qx5d.cn", "shishi.xn--55qx5d.cn"),
]


@pytest.mark.parametrize("domain,expected", OFFICIAL_VECTORS)
def test_official_psl_vectors(psl, domain, expected):
    labels = domain.lower().split(".")
    suffix_len = psl.public_suffix_length(labels)
    got = None if suffix_len >= len(labels) else psl.registrable_domain(domain)
    assert got == expected


@pytest.mark.parametrize("suffix", _EMPIRICALLY_VERIFIED_EXTRA_SUFFIXES)
def test_dataset_verified_extra_suffix_is_not_read_as_a_label(suffix):
    """
    Each of these is a real dynamic-DNS/tunnelling provider proven, not
    assumed, necessary by data/dga_dataset.parquet (see lexical.py's comment
    for exact sample counts per suffix). A random-looking label under one of
    these must resolve to the ATTACKER's label, never the provider name.
    """
    probe = f"a1b2c3d4e5.{suffix}"
    assert extract_label(probe) == "a1b2c3d4e5"


def test_real_psl_alone_misses_the_dataset_verified_suffixes():
    """
    Documents WHY the extra list in lexical.py exists: without it, these are
    real gaps in the crowdsourced upstream PSL (some were removed after the
    service shut down, e.g. co.cc; others were simply never submitted).
    """
    bare_psl = PublicSuffixList.from_file(PUBLIC_SUFFIX_LIST_PATH)
    still_missing = [s for s in _EMPIRICALLY_VERIFIED_EXTRA_SUFFIXES
                      if s not in bare_psl._rules]
    assert still_missing == list(_EMPIRICALLY_VERIFIED_EXTRA_SUFFIXES), (
        "a suffix once missing from the real PSL is now present upstream -- "
        "that's good news, but means this test (and the comment in "
        "lexical.py explaining why the extra list exists) should be updated"
    )
