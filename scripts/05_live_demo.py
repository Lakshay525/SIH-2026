"""
Quickest possible smoke test: loads both trained models and runs a handful
of hand-picked events through the real detection path
(src/pipeline/engine.process_event) -- the same function
scripts/07_stream_replay.py, scripts/08_benchmark.py, and dashboard.py use.
Run this after 02 and 04 just to sanity-check both models load and fire.

For an actual multi-source-IP streaming demo with persisted alerts and a
bounded-memory state manager, see scripts/06_generate_demo_stream.py +
scripts/07_stream_replay.py instead -- this script intentionally stays tiny.
"""
import sys
from pathlib import Path
import joblib

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import MODEL_DIR
from src.pipeline.state_manager import IPStateManager
from src.pipeline.engine import process_event


def main():
    dga_model = joblib.load(MODEL_DIR / "dga_lightgbm.pkl")
    tunnel_model = joblib.load(MODEL_DIR / "tunnelling_isolation_forest.pkl")
    state_mgr = IPStateManager()

    # -- DGA check on a few sample domains --
    for domain in ["google.com", "xqzplvmno-8f3ab21.net", "myshop24.in"]:
        alerts = process_event(
            {"ts": 0.0, "src_ip": "0.0.0.0", "domain": domain, "qtype": "A",
             "length": len(domain), "nxdomain": False},
            dga_model, tunnel_model, state_mgr,
        )
        dga_alerts = [a for a in alerts if a["threat_class"] == "DGA"]
        print(f"[DGA] {domain}: {'ALERT ' + str(dga_alerts[0]) if dga_alerts else 'clean'}")

    # -- Simulated live DNS stream for one source IP (exfil-style TXT burst) --
    tunnel_ip = "203.0.113.7"
    alert = None
    for i in range(45):
        domain = f"{i}-exfil.example.com"
        # Exfil-style lookups typically resolve to attacker-controlled nonexistent
        # names, so mark them NXDOMAIN -- this is what feeds nxdomain_rate.
        alerts = process_event(
            {"ts": float(i), "src_ip": tunnel_ip, "domain": domain, "qtype": "TXT",
             "length": len(domain), "nxdomain": True},
            dga_model, tunnel_model, state_mgr,
        )
        tunnel_alerts = [a for a in alerts if a["threat_class"] == "DNS_TUNNELLING"]
        if tunnel_alerts:
            alert = tunnel_alerts[0]
    print(f"\n[TUNNEL] {tunnel_ip}: {'ALERT ' + str(alert) if alert else 'clean'}")


if __name__ == "__main__":
    main()
