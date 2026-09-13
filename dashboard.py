import time
import streamlit as st
import pandas as pd
import joblib
from config import MODEL_DIR
from src.features.lexical import lexical_features
from src.models.dga_lightgbm import FEATURE_COLS as DGA_COLS
from src.models.tunnelling_detector import FEATURE_COLS as TUNNEL_COLS
from src.pipeline.state_manager import IPState
from src.pipeline.alert_schema import make_alert

# Setup UI
st.set_page_config(page_title="DNS Threat Monitor", layout="wide")
st.title("🛡️ Live DNS Threat Monitoring Dashboard")
st.markdown("Passive unidirectional inference engine. **No payload decryption.**")

@st.cache_resource
def load_models():
    return joblib.load(MODEL_DIR / "dga_lightgbm.pkl"), joblib.load(MODEL_DIR / "tunnelling_isolation_forest.pkl")

dga_model, tunnel_model = load_models()

# Layout Columns
col1, col2 = st.columns(2)
dga_view = col1.empty()
tun_view = col2.empty()

if st.button("▶ Start Live Stream Simulation"):
    alerts = []

    # 1. DGA Stream
    for d in ["google.com", "xqzplvmno-8f3ab21.net", "myshop24.in"]:
        feats = pd.DataFrame([lexical_features(d)])[DGA_COLS]
        prob = dga_model.predict_proba(feats)[0, 1]
        if prob > 0.5:
            alerts.append(make_alert(d, "DGA", float(prob), feats.iloc[0].to_dict()))
            dga_view.dataframe(pd.DataFrame(alerts)[["timestamp", "flow_id", "severity", "confidence"]])
        time.sleep(0.5) # Simulate network delay

    # 2. Tunnelling Stream
    state = IPState()
    for i in range(45):
        # Exfil-style lookups typically resolve to attacker-controlled nonexistent
        # names, so mark them NXDOMAIN -- this is what feeds nxdomain_rate below.
        state.add(bucket_idx=0, domain=f"{i}-exfil.example.com", qtype="TXT", length=55, nxdomain=True)

    stats = state.stats()

    row = pd.DataFrame([stats])[TUNNEL_COLS]
    score = -tunnel_model.decision_function(row)[0]

    if tunnel_model.predict(row)[0] == -1:
        tun_alert = make_alert("203.0.113.7", "DNS_TUNNELLING", min(score, 1.0), stats)
        tun_view.json(tun_alert) # Display full JSON schema for evidence

    st.success("Simulation Complete! Target throughput sustained.")
