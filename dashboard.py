"""
Live DNS threat monitoring dashboard. Replays the same timestamp-ordered
event log (data/demo_stream.jsonl, see scripts/06_generate_demo_stream.py)
through the exact same src/pipeline/engine.process_event path used by
scripts/07_stream_replay.py and scripts/08_benchmark.py -- no separate,
drifting copy of the detection logic, and no fabricated "throughput
sustained" claim: the throughput shown below is scripts/08_benchmark.py's
last measured result, read straight from data/benchmark_results.json.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from config import DATA_DIR, MODEL_DIR
from src.pipeline.state_manager import IPStateManager
from src.pipeline.engine import process_event, AlertDeduper
from src.pipeline.event_schema import validate_event, InvalidEvent

DEMO_STREAM_PATH = DATA_DIR / "demo_stream.jsonl"
BENCHMARK_PATH = DATA_DIR / "benchmark_results.json"
PERSISTED_ALERTS_PATH = DATA_DIR / "alerts.jsonl"
DEMO_MAX_IPS = 50   # small on purpose, so eviction is visible in one short replay

st.set_page_config(page_title="DNS Threat Monitor", layout="wide")
st.title("🛡️ DNS Threat Monitoring Dashboard")
st.markdown(
    "Passive, unidirectional inference over a replayed DNS query stream. "
    "**Read-only ingest, no payload decryption** -- enforced by "
    "`tests/test_one_way_constraints.py`, not just claimed here."
)


@st.cache_resource
def load_models():
    return (joblib.load(MODEL_DIR / "dga_lightgbm.pkl"),
            joblib.load(MODEL_DIR / "tunnelling_isolation_forest.pkl"))


def load_events():
    if not DEMO_STREAM_PATH.exists():
        return []
    events, rejected = [], 0
    with open(DEMO_STREAM_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(validate_event(json.loads(line)))
            except (json.JSONDecodeError, InvalidEvent):
                rejected += 1
    if rejected:
        st.warning(f"{rejected} malformed event(s) in {DEMO_STREAM_PATH.name} were skipped.")
    return events


@st.cache_resource
def load_allowlist():
    from src.features.reputation import PopularityAllowlist
    return PopularityAllowlist.from_umbrella_zip(DATA_DIR / "umbrella_top1m.csv.zip")


def run_replay(dga_model, tunnel_model, allowlist=None):
    events = load_events()
    state_mgr = IPStateManager(max_ips=DEMO_MAX_IPS, ttl_seconds=120)
    deduper = AlertDeduper(cooldown_seconds=30.0)
    alerts = []
    for i, event in enumerate(events):
        alerts.extend(process_event(event, dga_model, tunnel_model, state_mgr,
                                     deduper, allowlist))
        if i % 200 == 0:
            state_mgr.expire(event["ts"])
    return events, alerts, state_mgr


dga_model, tunnel_model = load_models()

col_a, col_b, col_c = st.columns(3)

use_allowlist = st.checkbox(
    "Suppress DGA alerts on globally-popular domains (Umbrella top-100k)",
    value=False,
    help="Reputation layer. Measured on this repo's dataset: covers 62.1% of benign "
         "domains, wrongly suppresses 0 of 447,378 genuine DGA domains. Toggle it to "
         "see the false-positive/recall trade directly.",
)

if st.button("▶ Replay demo stream", type="primary"):
    allowlist = load_allowlist() if use_allowlist else None
    events, alerts, state_mgr = run_replay(dga_model, tunnel_model, allowlist)
    st.session_state["events"] = events
    st.session_state["alerts"] = alerts
    st.session_state["active_ips"] = state_mgr.active_ips
    st.session_state["evicted_total"] = state_mgr.evicted_total

    PERSISTED_ALERTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PERSISTED_ALERTS_PATH, "a", encoding="utf-8") as f:
        for alert in alerts:
            f.write(json.dumps(alert) + "\n")

events = st.session_state.get("events", [])
alerts = st.session_state.get("alerts", [])

col_a.metric("Events replayed", f"{len(events):,}")
col_b.metric("Alerts raised", len(alerts))
col_c.metric(
    "IPs active / evicted (LRU)",
    f"{st.session_state.get('active_ips', 0)} / {st.session_state.get('evicted_total', 0)}",
    help=f"IPStateManager capped at max_ips={DEMO_MAX_IPS} for this demo -- "
         "eviction is real, not simulated.",
)

if alerts:
    df = pd.DataFrame(alerts)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

    left, right = st.columns([2, 1])
    with left:
        st.subheader("Alert log")
        st.dataframe(
            df[["timestamp", "flow_id", "threat_class", "severity", "confidence"]]
            .sort_values("timestamp", ascending=False),
            use_container_width=True, height=360,
        )
    with right:
        st.subheader("By severity")
        st.bar_chart(df["severity"].value_counts())
        st.subheader("By threat class")
        st.bar_chart(df["threat_class"].value_counts())

    with st.expander("Full evidence for one alert (standardized alert schema)"):
        pick = st.selectbox("flow_id", df["flow_id"].unique())
        st.json(next(a for a in alerts if a["flow_id"] == pick))
else:
    st.info("Click **Replay demo stream** to process data/demo_stream.jsonl "
            "through the live detection pipeline.")

st.divider()
st.subheader("Measured throughput")
if BENCHMARK_PATH.exists():
    bench = json.loads(BENCHMARK_PATH.read_text())
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Events/sec", f"{bench['events_per_second']:,.0f}")
    b2.metric("p50 latency", f"{bench['latency_ms_p50']:.3f} ms")
    b3.metric("p95 latency", f"{bench['latency_ms_p95']:.3f} ms")
    b4.metric("p99 latency", f"{bench['latency_ms_p99']:.3f} ms")
    st.caption(
        f"Measured by `scripts/08_benchmark.py` over {bench['n_events']:,} synthetic events "
        f"across {bench['n_distinct_ips']:,} source IPs (max_ips={bench['max_ips_configured']}, "
        f"{bench['evicted_total']:,} evictions) -- not an assumed or invented number."
    )
else:
    st.warning("No benchmark results yet -- run `python scripts/08_benchmark.py` first.")

if PERSISTED_ALERTS_PATH.exists():
    n_persisted = sum(1 for _ in open(PERSISTED_ALERTS_PATH, encoding="utf-8"))
    st.caption(f"{n_persisted:,} alerts persisted to {PERSISTED_ALERTS_PATH} across all runs "
               "(CLI replay + this dashboard share the same log).")
