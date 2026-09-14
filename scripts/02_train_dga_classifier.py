"""Train + evaluate the DGA classifier on config.DGA_DATASET_PATH."""
import json
import sys
import time
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DGA_DATASET_PATH, MODEL_DIR, DATA_DIR
from src.models.dga_lightgbm import FEATURE_COLS, train, evaluate, held_out_family_eval, calibration_report
from src.features.lexical import lexical_features
from sklearn.model_selection import train_test_split

def main():
    df = pd.read_parquet(DGA_DATASET_PATH)
    print(f"Loaded {len(df):,} domains, {df['family'].nunique()} labels (incl benign)")

    t0 = time.time()
    from tqdm import tqdm
    tqdm.pandas()

    # Extract features into a raw list first (with progress bar)
    raw_feats = df["domain"].progress_apply(lexical_features).tolist()

    # Convert the list to a DataFrame all at once (100x faster than pd.Series)
    feats = pd.DataFrame(raw_feats)
    df = pd.concat([df.reset_index(drop=True), feats.reset_index(drop=True)], axis=1)
    print(f"Feature extraction: {time.time()-t0:.1f}s")

    # Random split (optimistic number, keep for comparison only). This is
    # also the ONLY genuinely held-out data this run produces -- used below
    # for the calibration report, before it gets folded back into the final
    # production refit.
    train_idx, test_idx = train_test_split(df.index, test_size=0.2, stratify=df["label"], random_state=42)
    split_model = train(df, train_idx)
    report, auc = evaluate(split_model, df, test_idx)
    print(f"\n=== RANDOM SPLIT === recall={report['1']['recall']:.3f} "
          f"precision={report['1']['precision']:.3f} auc={auc:.3f}")

    # Calibration + operating-point report -- is predict_proba trustworthy,
    # and what does each severity cutoff (alert_schema.py) actually cost in
    # precision/recall? Computed from split_model's genuine 20% holdout,
    # before that data gets folded into the final refit below.
    calib = calibration_report(split_model, df, test_idx)
    print(f"\n=== CALIBRATION (Brier score, lower=better, 0=perfect) === {calib['brier_score']:.4f}")
    print("Severity thresholds (src/pipeline/alert_schema.py), measured not assumed:")
    for thresh, stats in calib["severity_thresholds"].items():
        print(f"  >={thresh:.2f}: precision={stats['precision']:.4f} "
              f"recall_at_or_above={stats['recall_at_or_above']:.3f} "
              f"n_flagged={stats['n_flagged']:,}")
    print("Operating points by false-positive budget:")
    for target_fp, pt in calib["operating_points_by_target_fp_rate"].items():
        if pt is None:
            print(f"  FP<={target_fp:.3f}: unreachable on this holdout")
        else:
            print(f"  FP<={target_fp:.3f}: threshold={pt['threshold']:.3f} "
                  f"precision={pt['precision']:.4f} recall={pt['recall']:.3f}")
    (DATA_DIR / "dga_calibration.json").write_text(json.dumps(calib, indent=2))

    # Held-out-family eval -- the honest generalization number. Also trains
    # each temporary model on df.index minus one family, same shape as the
    # random split above, so this doesn't touch the final refit either.
    families = sorted(df[df["family"] != "benign"]["family"].unique())
    if len(families) > 1:
        print("\n=== HELD-OUT FAMILY EVAL ===")
        recalls = {}
        for fam in families:
            rep, auc, _ = held_out_family_eval(df, fam)
            recalls[fam] = rep["1"]["recall"] if "1" in rep else float("nan")
            print(f"{fam:>20}: recall={recalls[fam]:.3f}")
        print(f"\nAverage held-out recall: {np.mean(list(recalls.values())):.3f}  <- report THIS number")
    else:
        print("\nOnly one DGA family present -- add more families for a held-out-family eval.")

    # Final production refit: every eval above used a model trained on only
    # 80% of the data (split_model) -- that's correct for getting an honest
    # held-out metric, but standard practice is to deploy a model refit on
    # ALL available data afterward, not leave 20% on the table in production.
    # The metrics above describe split_model's generalization behavior, which
    # this refit model should closely track (same features, same hyperparams,
    # ~25% more training data) without literally being the same fitted model.
    print("\nRefitting final model on the full dataset for deployment...")
    final_model = train(df, df.index)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODEL_DIR / "dga_lightgbm.pkl")
    print(f"\nSaved final model to {MODEL_DIR / 'dga_lightgbm.pkl'}")

if __name__ == "__main__":
    main()
