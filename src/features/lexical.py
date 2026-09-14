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
from config import WORDLIST_PATH

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

# Two-label public suffixes. Without these, "example.co.in" would have its
# registrable label read as "co" instead of "example" -- and .co.in/.net.in/
# .org.in/.ac.in are squarely in this project's actual deployment context, so
# this is a live failure mode, not a hypothetical one.
#
# Deliberately a bundled static list rather than `tldextract`: tldextract
# fetches the Public Suffix List over the network on first use, which would
# break the read-only/no-egress constraint this whole system is built around
# (tests/test_one_way_constraints.py would fail it). A static list is offline
# by construction. The trade-off is that it needs manual updating and doesn't
# cover every suffix in the full PSL -- see README.md Limitations.
_TWO_LABEL_SUFFIXES = frozenset("""
co.in net.in org.in firm.in gen.in ind.in ac.in edu.in res.in gov.in mil.in nic.in
co.uk org.uk me.uk ltd.uk plc.uk net.uk sch.uk ac.uk gov.uk nhs.uk police.uk mod.uk
com.au net.au org.au edu.au gov.au asn.au id.au
com.br net.br org.br gov.br edu.br
co.jp ne.jp or.jp ac.jp go.jp lg.jp
com.cn net.cn org.cn gov.cn edu.cn ac.cn
co.nz net.nz org.nz govt.nz ac.nz school.nz
co.za org.za net.za gov.za ac.za web.za
co.kr ne.kr or.kr re.kr go.kr ac.kr
com.mx org.mx gob.mx edu.mx net.mx
com.sg net.sg org.sg gov.sg edu.sg
com.hk net.hk org.hk gov.hk edu.hk
com.tw net.tw org.tw gov.tw edu.tw
co.th ac.th go.th in.th net.th or.th
co.id or.id ac.id go.id web.id net.id
com.my net.my org.my gov.my edu.my
com.ph net.ph org.ph gov.ph edu.ph
com.vn net.vn org.vn gov.vn edu.vn
com.pk com.bd com.np com.lk com.sa com.eg com.ng com.gh co.ke co.tz co.ug
com.tr com.ar com.co com.pe com.uy com.ve com.ec com.bo
co.il org.il net.il ac.il gov.il
com.ua net.ua org.ua com.pl net.pl org.pl gov.pl
com.ru net.ru org.ru com.es org.es edu.es gob.es
""".split())

# Delegation points where the PUBLIC registers subdomains -- dynamic-DNS
# providers, free-hosting and edge platforms. These are the PSL's "private
# section", and ignoring them is a real detection hole, not a nicety:
# 9.25% of the DGA domains in this repo's own dataset (41,382 of 447,378 --
# grandoreiro, g01, recjs, symmi, chaes, bamital, vidro, sutra ...) are
# dynamic-DNS names like "bf65a853.duckdns.org". Treating "duckdns.org" as a
# plain registrable domain would (a) feed the classifier the constant string
# "duckdns" instead of the attacker-generated label "bf65a853", and (b) let
# a popularity allowlist whitelist the whole provider, handing attackers a
# trivial, targeted bypass. The list below was derived from the parents
# actually present in data/dga_dataset.parquet, plus current platforms.
_PRIVATE_SUFFIXES = frozenset("""
ddns.net ddnsking.com duckdns.org dyndns.org dynalias.com dynserv.com dynu.net
dnsalias.com doesntexist.com endofinternet.net freedynamicdns.net freedynamicdns.org
gotdns.ch hopto.org isteingeek.de mooo.com myddns.me myftp.biz myftp.org mynumber.org
myvnc.com onthewifi.com redirectme.net servebbs.com servebeer.com serveblog.net
servecounterstrike.com serveftp.com servegame.com servehalflife.com servehttp.com
serveirc.com serveminecraft.net servemp3.com servepics.com servequake.com sytes.net
viewdns.net webhop.me webhop.info yi.org zapto.org bounceme.net 3utilities.com
no-ip.org no-ip.biz no-ip.info noip.me co.cc cz.cc
github.io pages.dev workers.dev localtunnel.me herokuapp.com blogspot.com
netlify.app vercel.app appspot.com firebaseapp.com azurewebsites.net
ngrok.io trycloudflare.com serveo.net glitch.me repl.co
""".split())

# One lookup for both kinds of suffix -- the algorithm is identical, only the
# reason for the entry differs.
_EFFECTIVE_SUFFIXES = _TWO_LABEL_SUFFIXES | _PRIVATE_SUFFIXES

# Long-tail fallback: the enumerated list above can't cover every country's
# second-level structure (.co.ls, .co.cy, .co.zm, .ac.at, ...), and the tail
# matters -- those were the last DGA domains still having their label read as
# the constant "co". A generic second-level label under a two-letter ccTLD is
# a registry suffix essentially everywhere it appears. The two-letter test is
# what keeps this safe: it can't fire on real domains like "go.com" or
# "co.com", because .com is not a ccTLD.
_GENERIC_SLDS = frozenset("""
co com net org edu gov ac mil gob go ne or re res firm gen ind nic web id asn
sch plc ltd me in govt school lg nom art rec med tur k12 biz info
""".split())


def _is_effective_suffix(second_level: str, tld: str) -> bool:
    candidate = f"{second_level}.{tld}"
    if candidate in _EFFECTIVE_SUFFIXES:
        return True
    return len(tld) == 2 and tld.isalpha() and second_level in _GENERIC_SLDS


def extract_label(domain: str) -> str:
    """
    Pick the label that actually carries DGA signal out of a possibly
    multi-level FQDN -- i.e. the registrable label (the "example" in
    example.com, example.co.uk, or cdn.example.co.in).

    DGA training data (DGArchive) is virtually always a bare `random.tld`
    (2 labels), so this is a no-op there. Real passively-observed traffic,
    though, routinely carries a CDN/tracking subdomain as the *leftmost*
    label (e.g. "a1b2c3d4.cdn.example.com") -- that label alone can look
    high-entropy/DGA-like by pure chance even though the query is completely
    benign. Reading the registrable label instead avoids that whole class of
    false positive.
    """
    parts = domain.rstrip(".").lower().split(".")
    if len(parts) < 2:
        return parts[0]
    # ".".join(parts[-2:]) is a known two-label public suffix (co.uk, co.in,
    # com.au, ...), so the registrable label is one position further left.
    if len(parts) >= 3 and _is_effective_suffix(parts[-2], parts[-1]):
        return parts[-3]
    return parts[-2]


def extract_registrable_domain(domain: str) -> str:
    """
    The registrable domain -- "example.com", "example.co.uk", "example.co.in".
    extract_label() returns just the "example" part; this keeps the suffix,
    which is what a popularity/reputation lookup has to match on (matching on
    the bare label alone would allowlist every domain that happens to share a
    label with a popular site).
    """
    parts = domain.rstrip(".").lower().split(".")
    if len(parts) < 2:
        return parts[0]
    if len(parts) >= 3 and _is_effective_suffix(parts[-2], parts[-1]):
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


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