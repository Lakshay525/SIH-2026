"""
End-to-end demo: loads both trained models, simulates a live DNS stream
through the state manager, and emits alerts in the standard schema.
Run this after 02 and 04.
"""
import sys
from pathlib import Path
import joblib
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import MODEL_DIR
from src.features.lexical import lexical_features
from src.models.dga_lightgbm import FEATURE_COLS as DGA_COLS
from src.models.tunnelling_detector import FEATURE_COLS as TUNNEL_COLS
from src.pipeline.state_manager import IPState
from src.pipeline.alert_schema import make_alert

def check_domain(dga_model, domain):
    feats = pd.DataFrame([lexical_features(domain)])[DGA_COLS]
    prob = dga_model.predict_proba(feats)[0, 1]
    if prob > 0.5:
        return make_alert(flow_id=domain, threat_class="DGA",
                           confidence=float(prob), evidence=feats.iloc[0].to_dict())
    return None

def check_ip_window(tunnel_model, ip, stats):
    # --- ADD THESE 3 LINES ---
    # Ensure any columns the model expects but the simulation missed are set to 0
    for col in TUNNEL_COLS:
        if col not in stats:
            stats[col] = 0.0
    # -------------------------

    row = pd.DataFrame([stats])[TUNNEL_COLS]
    score = -tunnel_model.decision_function(row)[0]
    flagged = tunnel_model.predict(row)[0] == -1
    
    if flagged:
        return make_alert(flow_id=ip, threat_class="DNS_TUNNELLING",
                          confidence=min(score, 1.0), evidence=stats)
    return None

def main():
    dga_model = joblib.load(MODEL_DIR / "dga_lightgbm.pkl")
    tunnel_model = joblib.load(MODEL_DIR / "tunnelling_isolation_forest.pkl")

    # -- DGA check on a few sample domains --
    for d in ["google.com", "xqzplvmno-8f3ab21.net", "myshop24.in"]:
        alert = check_domain(dga_model, d)
        print(f"[DGA] {d}: {'ALERT ' + str(alert) if alert else 'clean'}")

    # -- Simulated live DNS stream for one source IP --
    state = IPState()
    for i in range(45):
        state.add(bucket_idx=0, domain=f"{i}-exfil.example.com", qtype="TXT", length=55)
    alert = check_ip_window(tunnel_model, "203.0.113.7", state.stats())
    print(f"\n[TUNNEL] 203.0.113.7: {'ALERT ' + str(alert) if alert else 'clean'}")

if __name__ == "__main__":
    main()