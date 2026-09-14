"""DNS tunnelling detector -- Isolation Forest trained only on normal traffic windows."""
import numpy as np
from sklearn.ensemble import IsolationForest

FEATURE_COLS = ["query_rate", "unique_subdomains", "avg_query_len", "txt_ratio", "nxdomain_rate",
                 "non_a_ratio", "record_type_entropy"]


def as_matrix(windows):
    """
    Always hand sklearn a plain numpy matrix in FEATURE_COLS order.

    The live path (src/pipeline/engine.py) scores one event at a time from a
    raw numpy row -- building a one-row DataFrame per event was measurably
    expensive. Fitting on a DataFrame but predicting on arrays (or vice versa)
    makes sklearn emit "X does not have valid feature names" on every single
    call, so training and inference are kept numpy-consistent here instead.
    Column ORDER is the contract; FEATURE_COLS is its single definition.
    """
    if hasattr(windows, "columns"):
        return windows[FEATURE_COLS].to_numpy(dtype=np.float64)
    return np.asarray(windows, dtype=np.float64)


def train(normal_traffic_df, contamination=0.02, random_state=42):
    model = IsolationForest(contamination=contamination, random_state=random_state)
    model.fit(as_matrix(normal_traffic_df))
    return model


def score(model, windows_df):
    """Returns (anomaly_score, flagged). Higher score = more suspicious."""
    X = as_matrix(windows_df)
    # predict() is exactly sign(decision_function()) for IsolationForest, so
    # walk the forest once instead of twice.
    decision = model.decision_function(X)
    return -decision, decision < 0
