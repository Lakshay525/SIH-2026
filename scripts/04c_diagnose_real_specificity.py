"""
Why does the tunnelling detector flag ~90% of the REAL benign holdout?

scripts/04b reports the number; this script tests the explanation instead of
asserting it. Two competing hypotheses:

  H1 (construction artifact): the public dataset's "regular" class is a flat
      list of distinct domains, not one host's traffic log. Chunking it into
      fixed-size sessions forces unique_subdomains == query_rate, which no
      real host ever produces (real hosts revisit a small set of sites). The
      model is correctly rejecting a session shape that doesn't occur in life.

  H2 (miscalibrated training): the synthetic "normal" distribution in
      scripts/03 simply doesn't match real benign traffic, and the detector
      would false-positive at this rate on genuine host traffic too.

Test: hold the real data fixed and swap ONE feature at a time to its
synthetic-normal counterpart. If restoring a realistic revisit rate collapses
the false-positive rate, H1 is supported and the weakness is in the eval
harness. If the FP rate stays high, H2 is right and the detector is the
problem -- which is the more damaging answer, and the one worth knowing.

Reported honestly in README.md either way.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, MODEL_DIR
from src.models.tunnelling_detector import score, FEATURE_COLS

EVAL_PATH = DATA_DIR / "real_tunnel_eval_windows.parquet"
TRAIN_PATH = DATA_DIR / "tunnelling_windows.parquet"
OUT_PATH = DATA_DIR / "real_specificity_diagnosis.json"
SEED = 42


def fp_rate(model, df):
    _, flagged = score(model, df)
    return float(flagged.mean())


def main():
    if not EVAL_PATH.exists():
        print(f"{EVAL_PATH} not found -- run scripts/03b first.")
        return

    model = joblib.load(MODEL_DIR / "tunnelling_isolation_forest.pkl")
    real = pd.read_parquet(EVAL_PATH)
    benign = real[real["label"] == "normal"].reset_index(drop=True)
    train_normal = pd.read_parquet(TRAIN_PATH)
    train_normal = train_normal[train_normal["label"] == "normal"]

    baseline = fp_rate(model, benign)
    print(f"Real benign holdout sessions: {len(benign):,}")
    print(f"BASELINE false-positive rate: {baseline*100:.1f}%\n")

    print("Per-feature ablation -- replace ONE real feature with the synthetic")
    print("'normal' median, keep the other four real:\n")
    results = {}
    for col in FEATURE_COLS:
        patched = benign.copy()
        patched[col] = train_normal[col].median()
        rate = fp_rate(model, patched)
        results[col] = rate
        delta = (rate - baseline) * 100
        print(f"  {col:20s} FP={rate*100:6.1f}%   ({delta:+6.1f} pts vs baseline)")

    # The specific H1 claim: real hosts revisit sites, so unique_subdomains is
    # a small fraction of query_rate rather than equal to it.
    print("\nH1 test -- rebuild unique_subdomains with a realistic revisit rate")
    print("(every other feature stays exactly as measured from real traffic):\n")
    rng = np.random.default_rng(SEED)
    revisit = benign.copy()
    # A real host touches a handful of distinct names per minute, not a fresh
    # one per query; mirror the distribution scripts/03 models for normal hosts.
    revisit["unique_subdomains"] = np.minimum(
        rng.poisson(2.5, len(revisit)) + 1, revisit["query_rate"])
    revisit_rate = fp_rate(model, revisit)
    print(f"  unique_subdomains == query_rate (as built) : FP={baseline*100:6.1f}%")
    print(f"  unique_subdomains ~ realistic revisit rate : FP={revisit_rate*100:6.1f}%")

    verdict = ("H1 supported: the false-positive rate is dominated by the "
               "sessionisation artifact, not by the detector."
               if revisit_rate < baseline / 2 else
               "H1 NOT supported: the detector false-positives on real benign "
               "traffic even with a realistic revisit rate -- the synthetic "
               "'normal' distribution is genuinely miscalibrated.")
    print(f"\nVERDICT: {verdict}")

    OUT_PATH.write_text(json.dumps({
        "n_benign_sessions": len(benign),
        "baseline_fp_rate": baseline,
        "per_feature_ablation_fp_rate": results,
        "realistic_revisit_fp_rate": revisit_rate,
        "verdict": verdict,
    }, indent=2))
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
