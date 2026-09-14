"""
The actual "streaming, not batch" engine the PS requires: reads an ordered
JSONL event log (file or stdin, mirroring the port_scanning sibling branch's
`recon_detector stream` CLI convention) and processes it INCREMENTALLY, one
event at a time, in timestamp order -- not one big in-process loop over a
hardcoded list like the old scripts/05_live_demo.py.

  python scripts/07_stream_replay.py                        # replays data/demo_stream.jsonl
  python scripts/07_stream_replay.py --input path/to.jsonl
  cat events.jsonl | python scripts/07_stream_replay.py --input -

Every alert is appended to data/alerts.jsonl in the standardized schema
(src/pipeline/alert_schema.make_alert) as it's produced -- a persisted,
append-only log, which is what "clean chain of custody" actually requires
instead of alerts that vanish once printed.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import joblib

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, MODEL_DIR
from src.pipeline.state_manager import IPStateManager
from src.pipeline.engine import process_event, AlertDeduper
from src.features.reputation import PopularityAllowlist

DEFAULT_INPUT = DATA_DIR / "demo_stream.jsonl"
ALERTS_PATH = DATA_DIR / "alerts.jsonl"
EXPIRE_EVERY_N_EVENTS = 200
MAX_IPS = 50        # small on purpose for the demo -- makes eviction visible
TTL_SECONDS = 120


def read_events(path):
    fh = sys.stdin if path == "-" else open(path, encoding="utf-8")
    try:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)
    finally:
        if fh is not sys.stdin:
            fh.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=str(DEFAULT_INPUT),
                     help="JSONL event file, or '-' for stdin")
    ap.add_argument("--max-ips", type=int, default=MAX_IPS)
    ap.add_argument("--ttl-seconds", type=float, default=TTL_SECONDS)
    ap.add_argument("--allowlist", action="store_true",
                     help="suppress DGA alerts on globally-popular domains "
                          "(Umbrella top-100k) -- trades a little recall for a "
                          "lot of precision; run with and without to see both")
    args = ap.parse_args()

    dga_model = joblib.load(MODEL_DIR / "dga_lightgbm.pkl")
    tunnel_model = joblib.load(MODEL_DIR / "tunnelling_isolation_forest.pkl")
    state_mgr = IPStateManager(max_ips=args.max_ips, ttl_seconds=args.ttl_seconds)
    deduper = AlertDeduper(cooldown_seconds=30.0)

    allowlist = None
    if args.allowlist:
        allowlist = PopularityAllowlist.from_umbrella_zip(DATA_DIR / "umbrella_top1m.csv.zip")
        print(f"Popularity allowlist: {len(allowlist):,} registrable domains")

    ALERTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    n_events = 0
    n_alerts = 0
    latencies_ms = []
    wall_start = time.perf_counter()

    with open(ALERTS_PATH, "w", encoding="utf-8") as alerts_out:
        for event in read_events(args.input):
            t0 = time.perf_counter()
            alerts = process_event(event, dga_model, tunnel_model, state_mgr,
                                    deduper, allowlist)
            latencies_ms.append((time.perf_counter() - t0) * 1000)

            for alert in alerts:
                alerts_out.write(json.dumps(alert) + "\n")
                n_alerts += 1
                print(f"[{alert['severity']:8s}] {alert['threat_class']:15s} "
                      f"flow={alert['flow_id']} confidence={alert['confidence']:.3f}")

            n_events += 1
            # Expire idle IPs off *event* time, not wall-clock -- so a replay
            # is reproducible regardless of how fast this machine runs it.
            if n_events % EXPIRE_EVERY_N_EVENTS == 0:
                state_mgr.expire(event["ts"])

    wall_elapsed = time.perf_counter() - wall_start
    latencies_ms.sort()
    p95 = latencies_ms[int(0.95 * len(latencies_ms))] if latencies_ms else 0.0

    print(f"\n{n_events:,} events processed, {n_alerts} alerts raised, "
          f"in {wall_elapsed:.2f}s wall time ({n_events/wall_elapsed:,.0f} events/sec)")
    print(f"per-event latency: mean={sum(latencies_ms)/len(latencies_ms):.3f}ms "
          f"p95={p95:.3f}ms")
    print(f"IPStateManager: active_ips={state_mgr.active_ips} "
          f"evicted_total={state_mgr.evicted_total} (max_ips={args.max_ips})")
    print(f"Alerts written to {ALERTS_PATH}")


if __name__ == "__main__":
    main()
