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

def lexical_features(domain: str) -> dict:
    name = domain.split(".")[0]
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