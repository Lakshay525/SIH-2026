"""
Scores the trained tunnelling detector against the REALISTIC real-data
holdout (scripts/03c) -- benign sessions built with a realistic repeat-visit
pattern instead of scripts/03b's all-unique construction -- combined with the
real tunnel-tool sessions from scripts/03b (unchanged, since those already
reflect genuine tool behaviour). This is the honest answer to the question
scripts/04c's ablation raised but didn't resolve: with the identified
construction artifact actually fixed rather than just diagnosed, what
specificity does real domain data produce?

Still not a real per-host capture (see README "Real DNS packet ingestion"
for what that would take) -- the revisit PATTERN is simulated, only the
domain strings and their derived features are real -- but this directly
tests the artifact hypothesis instead of leaving it asserted.
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

REALISTIC_NORMAL_PATH = DATA_DIR / "real_tunnel_eval_realistic.parquet"
TUNNEL_SESSIONS_PATH = DATA_DIR / "real_tunnel_eval_windows.parquet"
OUT_PATH = DATA_DIR / "real_tunnel_eval_realistic_results.json"


def main():
    if not REALISTIC_NORMAL_PATH.exists():
        print(f"{REALISTIC_NORMAL_PATH} not found -- run scripts/03c first.")
        return
    if not TUNNEL_SESSIONS_PATH.exists():
        print(f"{TUNNEL_SESSIONS_PATH} not found -- run scripts/03b first.")
        return

    model = joblib.load(MODEL_DIR / "tunnelling_isolation_forest.pkl")
    realistic_normal = pd.read_parquet(REALISTIC_NORMAL_PATH)
    tunnel_sessions = pd.read_parquet(TUNNEL_SESSIONS_PATH)
    tunnel_only = tunnel_sessions[tunnel_sessions["label"] == "tunnel"]

    df = pd.concat([realistic_normal, tunnel_only], ignore_index=True)
    scores, flagged = score(model, df)
    y_true = (df["label"] != "normal").astype(int).to_numpy()
    y_pred = flagged.astype(int)

    report = classification_report(y_true, y_pred, digits=3, zero_division=0, output_dict=True)
    print("=== REALISTIC REAL-DATA HOLDOUT (repeat-visit sessions, real domain strings) ===")
    print(classification_report(y_true, y_pred, digits=3, zero_division=0))
    cm = confusion_matrix(y_true, y_pred)
    print("confusion matrix [[TN, FP], [FN, TP]]:")
    print(cm)

    print("\nPer-tool recall (tunnel tools only):")
    per_tool = {}
    for tool in sorted(tunnel_only["tool"].unique()):
        mask = (df["tool"] == tool).to_numpy()
        recall = y_pred[mask].mean()
        per_tool[tool] = float(recall)
        print(f"  {tool:8s}: recall={recall:.3f}  (n={mask.sum()})")

    normal_fp_rate = float(y_pred[(df["label"] == "normal").to_numpy()].mean())
    print(f"\nSpecificity on realistic real-derived normal sessions: "
          f"{1 - normal_fp_rate:.3f}  (FP rate: {normal_fp_rate:.3f})")
    print("Compare: scripts/04b's original (all-unique) construction gave 0% specificity "
          "(100% FP) on the same underlying domain pool.")

    results = {
        "n_normal_sessions": int((df["label"] == "normal").sum()),
        "n_tunnel_sessions": int((df["label"] == "tunnel").sum()),
        "normal_false_positive_rate": normal_fp_rate,
        "specificity": 1 - normal_fp_rate,
        "overall_report": report,
        "confusion_matrix": cm.tolist(),
        "per_tool_recall": per_tool,
        "note": (
            "Benign sessions here use realistic repeat-visit sampling (2-5 favourite "
            "real domains per synthetic host, sampled with replacement) instead of "
            "scripts/03b's all-unique construction -- see scripts/03c docstring. "
            "Still not a real per-host capture; the revisit PATTERN is simulated, "
            "only the domain strings and their derived features are real."
        ),
    }
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nSaved results to {OUT_PATH}")


if __name__ == "__main__":
    main()
