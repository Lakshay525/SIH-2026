"""DNS tunnelling detector -- Isolation Forest trained only on normal traffic windows."""
from sklearn.ensemble import IsolationForest

FEATURE_COLS = ["query_rate", "unique_subdomains", "avg_query_len", "txt_ratio", "nxdomain_rate"]

def train(normal_traffic_df, contamination=0.02, random_state=42):
    model = IsolationForest(contamination=contamination, random_state=random_state)
    model.fit(normal_traffic_df[FEATURE_COLS])
    return model

def score(model, windows_df):
    """Returns (anomaly_score, flagged). Higher score = more suspicious."""
    scores = -model.decision_function(windows_df[FEATURE_COLS])
    flagged = model.predict(windows_df[FEATURE_COLS]) == -1
    return scores, flagged