"""
Alerts must be stamped with when the traffic was OBSERVED, not when we
processed it.

This was a real bug: make_alert() used time.time() unconditionally and
engine.py never passed the event's own ts through, so replaying the same
captured stream twice produced two different sets of alert timestamps. For a
system whose whole value proposition is passive capture with a defensible
forensic timeline, that silently turns "when did the attack happen" into
"when did the analyst run the tool".
"""
import sys
import time
from pathlib import Path

import joblib
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import MODEL_DIR
from src.pipeline.alert_schema import make_alert
from src.pipeline.state_manager import IPStateManager
from src.pipeline.engine import process_event

# A capture from well in the past -- nothing here should ever equal "now".
CAPTURE_EPOCH = 1_600_000_000.0


def replay_events(n=120):
    events = []
    for i in range(n):
        events.append({
            "ts": CAPTURE_EPOCH + i * 0.5,
            "src_ip": "203.0.113.77" if i % 2 == 0 else f"10.0.0.{i % 15}",
            "domain": f"{i}chunkabcdef0123456789.tunnel.example.com" if i % 2 == 0
                      else f"site{i}.example.org",
            "qtype": "TXT" if i % 2 == 0 else "A",
            "length": 60 if i % 2 == 0 else 20,
            "nxdomain": i % 9 == 0,
        })
    return events


def run_once(dga_model, tunnel_model):
    state_mgr = IPStateManager(max_ips=50, ttl_seconds=120)
    alerts = []
    for event in replay_events():
        alerts.extend(process_event(event, dga_model, tunnel_model, state_mgr))
    return alerts


@pytest.fixture(scope="module")
def models():
    return (joblib.load(MODEL_DIR / "dga_lightgbm.pkl"),
            joblib.load(MODEL_DIR / "tunnelling_isolation_forest.pkl"))


def test_make_alert_uses_observed_timestamp():
    alert = make_alert("1.2.3.4", "DGA", 0.9, {}, observed_ts=CAPTURE_EPOCH)
    assert alert["timestamp"] == CAPTURE_EPOCH


def test_make_alert_falls_back_to_wall_clock_without_observed_ts():
    before = time.time()
    alert = make_alert("1.2.3.4", "DGA", 0.9, {})
    assert before <= alert["timestamp"] <= time.time()


def test_alert_timestamps_come_from_the_capture_not_the_clock(models):
    dga_model, tunnel_model = models
    alerts = run_once(dga_model, tunnel_model)
    assert alerts, "expected the synthetic tunnel burst to raise at least one alert"

    event_timestamps = {e["ts"] for e in replay_events()}
    now = time.time()
    for alert in alerts:
        assert alert["timestamp"] in event_timestamps, (
            "alert carries a timestamp that isn't any observed event's ts"
        )
        # Sanity: a 2020-era capture must never be stamped with today's clock.
        assert alert["timestamp"] < now - 86_400


def test_replaying_the_same_capture_twice_gives_identical_timestamps(models):
    """The actual chain-of-custody property: same input -> same timeline."""
    dga_model, tunnel_model = models
    first = run_once(dga_model, tunnel_model)
    time.sleep(0.05)  # guarantee wall-clock moved between the two runs
    second = run_once(dga_model, tunnel_model)

    assert [a["timestamp"] for a in first] == [a["timestamp"] for a in second]
    assert [a["flow_id"] for a in first] == [a["flow_id"] for a in second]
    assert [a["threat_class"] for a in first] == [a["threat_class"] for a in second]
    assert [a["confidence"] for a in first] == [a["confidence"] for a in second]
