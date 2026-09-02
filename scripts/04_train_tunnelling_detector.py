"""Train + evaluate the Isolation Forest tunnelling detector."""
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import TUNNEL_DATASET_PATH, MODEL_DIR
from src.models.tunnelling_detector import train, score, FEATURE_COLS

def main():
    df = pd.read_parquet(TUNNEL_DATASET_PATH)
    normal = df[df["label"] == "normal"]
    other = df[df["label"] != "normal"]

    print(f"Training on {len(normal)} normal windows")
    model = train(normal)

    all_windows = pd.concat([normal, other], ignore_index=True)
    y_true = np.array([0] * len(normal) + [1] * len(other))
    scores, flagged = score(model, all_windows)
    print(classification_report(y_true, flagged.astype(int), digits=3, zero_division=0))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "tunnelling_isolation_forest.pkl")
    print(f"Saved model to {MODEL_DIR / 'tunnelling_isolation_forest.pkl'}")

if __name__ == "__main__":
    main()