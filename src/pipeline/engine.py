"""
Single shared per-event processing path: feature extraction -> model
inference -> alerting. Every place that touches live/replayed traffic
(scripts/07_stream_replay.py, scripts/08_benchmark.py, dashboard.py) calls
`process_event` instead of re-implementing this logic -- previously the DGA
check + tunnelling check + patch-to-zero workaround were copy-pasted across
scripts/05_live_demo.py and dashboard.py and had already drifted (the
nxdomain_rate bug lived in both copies).

An `event` is a plain dict describing one observed DNS query, exactly what a
passive mirror/tap can see -- nothing here ever resolves, connects, or probes
anything (see tests/test_one_way_constraints.py):
    {
        "ts": float,        # event/capture timestamp, seconds
        "src_ip": str,      # source IP the query came from
        "domain": str,      # queried name
        "qtype": str,       # DNS record type, e.g. "A", "TXT", "NULL", "CNAME"
        "length": int,      # length of the query name in bytes/chars
        "nxdomain": bool,   # whether the response (if observed) was NXDOMAIN
    }
"""
import pandas as pd

from src.features.lexical import lexical_features
from src.models.dga_lightgbm import FEATURE_COLS as DGA_COLS
from src.models.tunnelling_detector import FEATURE_COLS as TUNNEL_COLS
from src.pipeline.alert_schema import make_alert

DGA_THRESHOLD = 0.5
# A window with only a handful of queries so far is a thin, incomplete
# sample -- an Isolation Forest trained on ~60s-worth of typical traffic
# will call it "unusual" purely for having too little data, not because it
# looks like tunnelling. Waiting for a minimum sample size before scoring
# cuts that transient noise almost for free: real tunnelling sessions push
# 30-90+ queries into one window (see scripts/03), so this barely delays
# genuine detection while suppressing false alarms on brand-new/quiet IPs.
MIN_QUERIES_FOR_TUNNEL_SCORING = 5


def _pyfloat(x) -> float:
    """Strip numpy scalar types so alerts stay plain-JSON-serializable."""
    return float(x)


class AlertDeduper:
    """
    Suppresses repeat (threat_class, flow_id) alerts within a cooldown window.
    Without this, an ongoing tunnelling session re-triggers a fresh alert on
    every single subsequent event (dozens of identical pages for one incident)
    -- a real alerting pipeline pages once per ongoing incident, not once per
    packet. Optional: pass deduper=None to process_event to disable this and
    see every raw firing (e.g. useful for benchmarking raw detection rate).
    """
    def __init__(self, cooldown_seconds: float = 30.0):
        self.cooldown_seconds = cooldown_seconds
        self._last_alert_ts = {}

    def should_emit(self, threat_class: str, flow_id: str, ts: float) -> bool:
        key = (threat_class, flow_id)
        last = self._last_alert_ts.get(key)
        if last is not None and ts - last < self.cooldown_seconds:
            return False
        self._last_alert_ts[key] = ts
        return True


def check_dga(dga_model, event: dict):
    domain = event["domain"]
    feats = pd.DataFrame([lexical_features(domain)])[DGA_COLS]
    prob = _pyfloat(dga_model.predict_proba(feats)[0, 1])
    if prob <= DGA_THRESHOLD:
        return None
    evidence = {k: _pyfloat(v) for k, v in feats.iloc[0].to_dict().items()}
    evidence["src_ip"] = event.get("src_ip")
    return make_alert(flow_id=domain, threat_class="DGA", confidence=prob, evidence=evidence)


def check_tunnelling(tunnel_model, state_mgr, event: dict):
    ip = event["src_ip"]
    state_mgr.add_event(ip, event["ts"], event["domain"], event["qtype"],
                         event["length"], event.get("nxdomain", False))
    stats = state_mgr.get_stats(ip, current_ts=event["ts"])
    if stats["query_rate"] < MIN_QUERIES_FOR_TUNNEL_SCORING:
        return None
    row = pd.DataFrame([stats])[TUNNEL_COLS]
    score = _pyfloat(-tunnel_model.decision_function(row)[0])
    flagged = tunnel_model.predict(row)[0] == -1
    if not flagged:
        return None
    evidence = {k: _pyfloat(v) for k, v in stats.items()}
    return make_alert(flow_id=ip, threat_class="DNS_TUNNELLING",
                       confidence=min(score, 1.0), evidence=evidence)


def process_event(event: dict, dga_model, tunnel_model, state_mgr, deduper=None) -> list:
    """
    Run one observed DNS query through both detectors. Returns 0-2 alerts.
    Pass an AlertDeduper to collapse an ongoing incident's repeat firings into
    one alert per cooldown window instead of one per event.
    """
    alerts = []
    dga_alert = check_dga(dga_model, event)
    if dga_alert:
        alerts.append(dga_alert)
    tunnel_alert = check_tunnelling(tunnel_model, state_mgr, event)
    if tunnel_alert:
        alerts.append(tunnel_alert)

    if deduper is not None:
        alerts = [a for a in alerts
                  if deduper.should_emit(a["threat_class"], a["flow_id"], event["ts"])]
    return alerts
