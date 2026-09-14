"""
calibration_report() answers two questions the training script used to never
ask: is predict_proba actually trustworthy as a probability (Brier score),
and what does each existing severity cutoff (0.30/0.60/0.85, previously just
round numbers) cost in real precision/recall.
"""
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DGA_DATASET_PATH, MODEL_DIR
from src.features.lexical import lexical_features
from src.models.dga_lightgbm import FEATURE_COLS, calibration_report


@pytest.fixture(scope="module")
def small_labeled_sample():
    """A small, fast-to-featurize sample -- not a rigorous eval, just enough
    to exercise calibration_report()'s shape and sanity-check its output."""
    df = pd.read_parquet(DGA_DATASET_PATH)
    sample = df.sample(n=2000, random_state=7).reset_index(drop=True)
    feats = pd.DataFrame(sample["domain"].map(lexical_features).tolist())
    return pd.concat([sample, feats], axis=1)


@pytest.fixture(scope="module")
def dga_model():
    return joblib.load(MODEL_DIR / "dga_lightgbm.pkl")


def test_calibration_report_shape(dga_model, small_labeled_sample):
    report = calibration_report(dga_model, small_labeled_sample, small_labeled_sample.index)

    assert 0.0 <= report["brier_score"] <= 1.0
    assert set(report["severity_thresholds"].keys()) == {0.30, 0.60, 0.85}
    for stats in report["severity_thresholds"].values():
        assert 0.0 <= stats["precision"] <= 1.0
        assert 0.0 <= stats["recall_at_or_above"] <= 1.0
        assert stats["n_flagged"] >= 0

    assert set(report["operating_points_by_target_fp_rate"].keys()) == \
        {0.10, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001}


def test_precision_increases_with_severity(dga_model, small_labeled_sample):
    """The whole point of the severity tiers: higher confidence should mean
    higher precision. Not guaranteed by construction -- this is what
    calibration_report() actually verifies rather than assumes."""
    report = calibration_report(dga_model, small_labeled_sample, small_labeled_sample.index)
    p_medium = report["severity_thresholds"][0.30]["precision"]
    p_high = report["severity_thresholds"][0.60]["precision"]
    p_critical = report["severity_thresholds"][0.85]["precision"]
    assert p_medium <= p_high <= p_critical


def test_operating_points_are_json_serializable(dga_model, small_labeled_sample):
    import json
    report = calibration_report(dga_model, small_labeled_sample, small_labeled_sample.index)
    json.dumps(report)  # must not raise -- no numpy scalar types leaking through
