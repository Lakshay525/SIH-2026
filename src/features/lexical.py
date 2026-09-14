"""
Turns a domain-name string into numeric features for the DGA classifier.
ngram_score + dict_word_ratio are the features that actually make it
generalize to unseen DGA families -- length/entropy alone don't.
"""
import math
from collections import Counter
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import WORDLIST_PATH, PUBLIC_SUFFIX_LIST_PATH
from src.features.public_suffix import PublicSuffixList

VOWELS = set("aeiou")

# Small built-in fallback so the code runs even before you've dropped in a
# real word list. Swap WORDLIST_PATH to any real dictionary file (one word
# per line) for much better accuracy -- e.g. /usr/share/dict/words on
# Linux/Mac, or any wordlist off GitHub.
_FALLBACK_WORDS = [
    "cloud","tech","data","secure","bank","market","shop","news","media","health",
    "school","travel","food","music","game","sport","finance","study","learn","connect",
    "global","smart","digital","online","service","group","system","network","solutions",
    "trade","city","home","life","world","team","hub","zone","express","direct",
]

def _load_words():
    if WORDLIST_PATH.exists():
        with open(WORDLIST_PATH) as f:
            return [w.strip().lower() for w in f if w.strip()]
    return _FALLBACK_WORDS

ENGLISH_WORDS = _load_words()
ENGLISH_WORDS_SET = {w for w in ENGLISH_WORDS if len(w) >= 3}

def _build_bigram_model(words):
    counts, total = Counter(), 0
    for w in words:
        for i in range(len(w) - 1):
            counts[w[i:i+2]] += 1
            total += 1
    return {k: v / total for k, v in counts.items()} if total else {}

BIGRAM_MODEL = _build_bigram_model(ENGLISH_WORDS)
MIN_PROB = 1e-6


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())

def digit_ratio(s: str) -> float:
    return sum(ch.isdigit() for ch in s) / max(len(s), 1)

def vowel_consonant_ratio(s: str) -> float:
    letters = [c for c in s if c.isalpha()]
    v = sum(c in VOWELS for c in letters)
    c = len(letters) - v
    return v / max(c, 1)

def max_consonant_run(s: str) -> int:
    run = max_run = 0
    for ch in s:
        if ch.isalpha() and ch not in VOWELS:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return max_run

def ngram_score(name: str) -> float:
    name = name.lower()
    if len(name) < 2:
        return 0.0
    scores = [math.log(BIGRAM_MODEL.get(name[i:i+2], MIN_PROB)) for i in range(len(name) - 1)]
    return sum(scores) / len(scores)

def dictionary_word_ratio(name: str) -> float:
    name = name.lower()
    n = len(name)
    covered = [False] * n
    for i in range(n):
        for j in range(i + 3, min(i + 13, n + 1)):
            if name[i:j] in ENGLISH_WORDS_SET:
                for k in range(i, j):
                    covered[k] = True
    return sum(covered) / max(n, 1)

def hex_char_ratio(s: str) -> float:
    hex_chars = set("0123456789abcdef")
    return sum(c in hex_chars for c in s.lower()) / max(len(s), 1)

# Real Mozilla Public Suffix List, parsed offline from a vendored snapshot --
# see src/features/public_suffix.py for the algorithm and why this is a
# static file rather than a live-fetching library like `tldextract` (which
# would break the read-only/no-egress constraint this system is built
# around; tests/test_one_way_constraints.py fails that). Verified against
# all 73 applicable vectors in the PSL project's own official test suite.
#
# This natively covers everything a hand-maintained list used to have to
# enumerate by hand -- including a real, previously-hidden detection hole:
# 9.25% of the DGA domains in this repo's own dataset (41,382 of 447,378 --
# grandoreiro, g01, recjs, symmi, chaes, bamital, vidro, sutra ...) are
# dynamic-DNS names like "bf65a853.duckdns.org", where the registrable label
# is "bf65a853" (the attacker's label), not "duckdns" (the provider) --
# "duckdns.org" is correctly in the PSL's PRIVATE section.
#
# The official PSL is the primary source (verified against all 73 applicable
# vectors in its own test suite), but it's crowdsourced and doesn't list
# every provider -- checking it against every dynamic-DNS/hosting delegation
# point in the original hand-maintained list this replaced found 13 real
# gaps. Each one either appears as the constant parent of real DGA domains in
# data/dga_dataset.parquet (mynumber.org: 1,670 samples; localtunnel.me: 129
# samples, family=symmi; co.cc/cz.cc: 1,262/1,241 samples -- once in the PSL,
# removed after the service shut down years ago, but this dataset's DGA
# samples still use them) or is a live tunnelling/hosting platform in the
# same category as ones the PSL does list (serveo.net, glitch.me). Omitting
# any of the dataset-verified ones re-opens the detection hole above for
# those specific families.
_EMPIRICALLY_VERIFIED_EXTRA_SUFFIXES = (
    "dynserv.com freedynamicdns.net freedynamicdns.org mooo.com myddns.me "
    "mynumber.org viewdns.net yi.org co.cc cz.cc localtunnel.me "
    "serveo.net glitch.me"
).split()

_PSL = PublicSuffixList.from_file(PUBLIC_SUFFIX_LIST_PATH,
                                  extra_rules=_EMPIRICALLY_VERIFIED_EXTRA_SUFFIXES)


def extract_label(domain: str) -> str:
    """
    Pick the label that actually carries DGA signal out of a possibly
    multi-level FQDN -- i.e. the registrable label (the "example" in
    example.com, example.co.uk, cdn.example.co.in, or the "bf65a853" in
    bf65a853.duckdns.org).

    DGA training data (DGArchive) is virtually always a bare `random.tld`
    (2 labels), so this is a no-op there. Real passively-observed traffic,
    though, routinely carries a CDN/tracking subdomain as the *leftmost*
    label (e.g. "a1b2c3d4.cdn.example.com") -- that label alone can look
    high-entropy/DGA-like by pure chance even though the query is completely
    benign. Reading the registrable label instead avoids that whole class of
    false positive.
    """
    return _PSL.registrable_label(domain)


def extract_registrable_domain(domain: str) -> str:
    """
    The registrable domain -- "example.com", "example.co.uk",
    "bf65a853.duckdns.org". extract_label() returns just the leftmost part;
    this keeps the suffix, which is what a popularity/reputation lookup has
    to match on (matching on the bare label alone would allowlist every
    domain that happens to share a label with a popular site; matching on
    just the provider for a dynamic-DNS name would allowlist every C2
    hosted there).
    """
    return _PSL.registrable_domain(domain)


def lexical_features(domain: str) -> dict:
    name = extract_label(domain)
    return {
        "length": len(name),
        "entropy": shannon_entropy(name),
        "digit_ratio": digit_ratio(name),
        "vowel_consonant_ratio": vowel_consonant_ratio(name),
        "max_consonant_run": max_consonant_run(name),
        "ngram_score": ngram_score(name),
        "dict_word_ratio": dictionary_word_ratio(name),
        "hex_char_ratio": hex_char_ratio(name),
    }