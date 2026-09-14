"""
A SECOND, more realistic real-data holdout for the "regular" (benign) class
only -- built after scripts/04c's ablation identified exactly why the
original scripts/03b construction produces a 100% false-positive rate on
real benign sessions: it groups the Mendeley dataset's "regular" class (a
flat, distinct list of popular domains, like a top-domain list) into
fixed-size chunks, which forces unique_subdomains == query_rate. No real
host does that -- a real host revisits a small personal set of sites
repeatedly.

This script keeps every value that IS real (the domain strings, hence their
real lengths/qtypes/entropy) but fixes the ONE identified construction
artifact: each synthetic "host" gets a small personal set of 2-5 favourite
domains (sampled from the real regular-domain pool) and its session is built
by sampling WITH REPLACEMENT from that small set -- mimicking a real host's
repeat-visit behaviour, using 100% real domain strings throughout.

This is still not a real capture (the *pattern* of revisits is simulated,
not observed) -- but it directly tests scripts/04c's own diagnosis rather
than leaving it as an untested hypothesis. Tunnel-tool sessions are
unchanged from scripts/03b: real tunnelling sessions genuinely are
near-100%-unique per query (that's accurate behaviour, not an artifact), so
there's nothing to fix there.
"""
import math
import sys
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR

RAW_CSV = DATA_DIR / "real_tunnel_domains.csv"
OUT_PATH = DATA_DIR / "real_tunnel_eval_realistic.parquet"
A_QTYPE = 1
TXT_QTYPE = 16

N_SESSIONS = 416          # match scripts/03b's real-normal session count, for a fair comparison
SESSION_SIZE_RANGE = (8, 20)
FAVOURITES_RANGE = (2, 5)  # a real host revisits a small personal set, not one, not hundreds
SEED = 42


def record_type_entropy(qtypes) -> float:
    counts = Counter(qtypes)
    total = sum(counts.values())
    return -sum((c / total) * math.log2(c / total) for c in counts.values() if c)


def build_realistic_normal_sessions(regular: pd.DataFrame, rng) -> pd.DataFrame:
    windows = []
    pool_idx = regular.index.to_numpy()
    for _ in range(N_SESSIONS):
        n_favourites = rng.integers(*FAVOURITES_RANGE, endpoint=True)
        favourites = regular.loc[rng.choice(pool_idx, size=n_favourites, replace=False)]
        session_size = int(rng.integers(*SESSION_SIZE_RANGE, endpoint=True))
        # Sample WITH replacement from the small favourites set -- this is
        # the fix: a real host repeats visits, it doesn't issue a fresh
        # unique query every time.
        chunk = favourites.sample(n=session_size, replace=True, random_state=int(rng.integers(1e9)))
        windows.append({
            "query_rate": len(chunk),
            "unique_subdomains": chunk["qname"].nunique(),
            "avg_query_len": chunk["qd_qname_len"].mean(),
            "txt_ratio": (chunk["qd_qtype"] == TXT_QTYPE).mean(),
            "nxdomain_rate": 0.0,
            "non_a_ratio": (chunk["qd_qtype"] != A_QTYPE).mean(),
            "record_type_entropy": record_type_entropy(chunk["qd_qtype"]),
            "label": "normal",
            "tool": "normal_realistic",
        })
    return pd.DataFrame(windows)


def main():
    if not RAW_CSV.exists():
        print(f"{RAW_CSV} not found -- run this after downloading the real dataset.")
        return

    df = pd.read_csv(RAW_CSV)
    regular = df[df["label"] == 0]
    rng = np.random.default_rng(SEED)

    out = build_realistic_normal_sessions(regular, rng)
    out.to_parquet(OUT_PATH)

    print(f"Built {len(out):,} realistic real-derived normal sessions from "
          f"{len(regular):,} real regular domains")
    print(out[["query_rate", "unique_subdomains", "avg_query_len",
                "txt_ratio", "non_a_ratio", "record_type_entropy"]].describe().loc[
        ["mean", "min", "max"]
    ])
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
