"""DGA classifier -- LightGBM on lexical features."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import brier_score_loss, classification_report, precision_recall_curve, roc_auc_score

from src.features.lexical import lexical_features

FEATURE_COLS = [
    "length", "entropy", "digit_ratio", "vowel_consonant_ratio",
    "max_consonant_run", "ngram_score", "dict_word_ratio", "hex_char_ratio",
]

def build_feature_frame(domains, labels, families=None):
    df = pd.DataFrame({"domain": domains, "label": labels})
    if families is not None:
        df["family"] = families
    feats = df["domain"].apply(lexical_features).apply(pd.Series)
    return pd.concat([df, feats], axis=1)

def train(df, train_idx):
    model = LGBMClassifier(n_estimators=200, class_weight="balanced", verbosity=-1)
    model.fit(df.loc[train_idx, FEATURE_COLS], df.loc[train_idx, "label"])
    return model

def evaluate(model, df, test_idx):
    X_test = df.loc[test_idx, FEATURE_COLS]
    y_test = df.loc[test_idx, "label"]
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, preds, digits=3, output_dict=True, zero_division=0)
    auc = roc_auc_score(y_test, probs) if len(set(y_test)) > 1 else float("nan")
    return report, auc

def predict_batch(model, feature_matrix: np.ndarray) -> np.ndarray:
    """
    Score many rows in one call via the raw booster -- see engine.py's
    check_dga for why this matters: a fresh one-row pandas DataFrame per
    event plus the sklearn predict_proba wrapper cost ~4.3ms/event; the
    booster on a raw numpy row costs ~0.4ms; the booster on a BATCH of rows
    amortizes to ~0.004ms/row (measured: 300 rows in 1.29ms) -- ~100x the
    single-event path. Real DNS collectors typically buffer micro-batches
    (e.g. 100-1000 queries every ~100ms) rather than emit one query at a
    time, so this is the throughput path a real deployment would actually
    use; the single-event path (engine.py) stays as the lowest-latency
    option for a strict one-at-a-time stream.

    feature_matrix: shape (n, len(FEATURE_COLS)), columns in FEATURE_COLS
    order -- callers build this themselves (e.g. via lexical_features per
    row) rather than this function reaching back into raw domains, so it
    has no opinion on where the rows came from.
    """
    return model.booster_.predict(np.asarray(feature_matrix, dtype=np.float64))


def calibration_report(model, df, test_idx, severity_thresholds=(0.30, 0.60, 0.85)):
    """
    Is predict_proba trustworthy as a probability, and what does each
    existing severity boundary (src/pipeline/alert_schema.py) actually cost
    in precision/recall? Answered by measurement, not assumed -- the
    severity cutoffs (0.30/0.60/0.85) were originally just round numbers.
    """
    X = df.loc[test_idx, FEATURE_COLS]
    y = df.loc[test_idx, "label"].to_numpy()
    probs = model.predict_proba(X)[:, 1]

    brier = brier_score_loss(y, probs)
    precision, recall, thresholds = precision_recall_curve(y, probs)

    severity_report = {}
    for thresh in severity_thresholds:
        i = int(np.searchsorted(thresholds, thresh))
        severity_report[thresh] = {
            "precision": float(precision[i]),
            "recall_at_or_above": float(recall[i]),
            "n_flagged": int((probs >= thresh).sum()),
        }

    # A concrete operating-point menu for a real deployment decision, not
    # just the three existing severity cutoffs -- e.g. "what threshold keeps
    # false positives under 1%".
    operating_points = {}
    for target_fp in (0.10, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001):
        ok = np.where((1 - precision[:-1]) <= target_fp)[0]
        if len(ok):
            i = int(ok[np.argmax(recall[:-1][ok])])
            operating_points[target_fp] = {
                "threshold": float(thresholds[i]),
                "precision": float(precision[i]),
                "recall": float(recall[i]),
            }
        else:
            operating_points[target_fp] = None

    return {
        "brier_score": float(brier),
        "severity_thresholds": severity_report,
        "operating_points_by_target_fp_rate": operating_points,
    }


def held_out_family_eval(df, held_out_family, benign_sample_frac=0.3, random_state=1):
    """Honest generalization test: train on every family except one, test on that one."""
    train_idx = df[df["family"] != held_out_family].index
    benign_test_idx = df[df["family"] == "benign"].sample(
        frac=benign_sample_frac, random_state=random_state
    ).index
    dga_test_idx = df[df["family"] == held_out_family].index
    test_idx = benign_test_idx.union(dga_test_idx)

    model = train(df, train_idx)
    report, auc = evaluate(model, df, test_idx)
    return report, auc, model