"""
Honest generalization check: score the tunnelling Isolation Forest (trained
purely on the protocol-realistic synthetic set from scripts/03/04) against
the REAL-DATA holdout windows built by scripts/03b. This data was never seen
during training -- this is the number to actually report, not the synthetic
self-eval printed by scripts/04.
"""
import json
import sys
from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, MODEL_DIR
from src.models.tunnelling_detector import score, FEATURE_COLS

EVAL_PATH = DATA_DIR / "real_tunnel_eval_windows.parquet"
OUT_PATH = DATA_DIR / "real_tunnel_eval.json"


def main():
    if not EVAL_PATH.exists():
        print(f"{EVAL_PATH} not found -- run scripts/03b_build_real_tunnel_holdout.py first.")
        return

    model = joblib.load(MODEL_DIR / "tunnelling_isolation_forest.pkl")
    df = pd.read_parquet(EVAL_PATH)

    scores, flagged = score(model, df)
    y_true = (df["label"] != "normal").astype(int).to_numpy()
    y_pred = flagged.astype(int)

    report = classification_report(y_true, y_pred, digits=3, zero_division=0, output_dict=True)
    print("=== REAL-DATA HOLDOUT (never trained on) ===")
    print(classification_report(y_true, y_pred, digits=3, zero_division=0))
    print("confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_true, y_pred))

    print("\nPer-tool recall (tunnel tools only):")
    per_tool = {}
    for tool in sorted(df.loc[df["label"] == "tunnel", "tool"].unique()):
        mask = df["tool"] == tool
        recall = y_pred[mask.to_numpy()].mean()
        per_tool[tool] = float(recall)
        print(f"  {tool:8s}: recall={recall:.3f}  (n={mask.sum()})")

    results = {
        "n_windows": len(df),
        "overall_report": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "per_tool_recall": per_tool,
        "note": (
            "txt_ratio is 0.0 for real iodine/tuns sessions in this dataset "
            "(they use NULL/CNAME, not TXT) -- detection for those tools, if "
            "it works, is being driven by avg_query_len, not txt_ratio. "
            "nxdomain_rate is fixed at 0.0 for all real-derived sessions "
            "(not derivable from this dataset -- see scripts/03b docstring)."
        ),
    }
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nSaved results to {OUT_PATH}")


if __name__ == "__main__":
    main()
