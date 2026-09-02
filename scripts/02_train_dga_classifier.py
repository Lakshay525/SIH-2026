"""Train + evaluate the DGA classifier on config.DGA_DATASET_PATH."""
import sys
import time
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DGA_DATASET_PATH, MODEL_DIR
from src.models.dga_lightgbm import FEATURE_COLS, train, evaluate, held_out_family_eval
from src.features.lexical import lexical_features
from sklearn.model_selection import train_test_split

def main():
    df = pd.read_parquet(DGA_DATASET_PATH)
    print(f"Loaded {len(df):,} domains, {df['family'].nunique()} labels (incl benign)")

    t0 = time.time()
    feats = df["domain"].apply(lexical_features).apply(pd.Series)
    df = pd.concat([df.reset_index(drop=True), feats.reset_index(drop=True)], axis=1)
    print(f"Feature extraction: {time.time()-t0:.1f}s")

    # Random split (optimistic number, keep for comparison only)
    train_idx, test_idx = train_test_split(df.index, test_size=0.2, stratify=df["label"], random_state=42)
    model = train(df, train_idx)
    report, auc = evaluate(model, df, test_idx)
    print(f"\n=== RANDOM SPLIT === recall={report['1']['recall']:.3f} "
          f"precision={report['1']['precision']:.3f} auc={auc:.3f}")

    # Held-out-family eval -- the honest generalization number
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

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "dga_lightgbm.pkl")
    print(f"\nSaved final model to {MODEL_DIR / 'dga_lightgbm.pkl'}")

if __name__ == "__main__":
    main()
    