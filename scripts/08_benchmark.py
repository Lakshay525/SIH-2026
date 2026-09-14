"""
Throughput benchmark -- the PS explicitly requires stating and demonstrating
the traffic rate this was tested against. Modeled on the port_scanning
sibling branch's evaluation/benchmark.py: time every event individually with
time.perf_counter(), report MEASURED numbers only (never invented), and prove
memory stayed bounded under load by tracking active_ips/evicted_total on a
deliberately small max_ips so eviction actually has to fire.

    python scripts/08_benchmark.py [--n 50000] [--n-ips 2000] [--max-ips 500]
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import joblib

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, MODEL_DIR
from src.pipeline.state_manager import IPStateManager
from src.pipeline.engine import process_event, check_dga, check_dga_batch

OUT_PATH = DATA_DIR / "benchmark_results.json"


def percentile(sorted_vals, p):
    if not sorted_vals:
        return 0.0
    idx = min(int(p * len(sorted_vals)), len(sorted_vals) - 1)
    return sorted_vals[idx]


def gen_events(n, n_ips, seed=123):
    rng = random.Random(seed)
    events = []
    for i in range(n):
        ip = f"10.0.{(i % n_ips) // 254}.{(i % n_ips) % 254 + 1}"
        ts = i * 0.01  # 100 events/sec of synthetic wall-clock-equivalent time
        domain = f"{rng.randrange(10**8, 10**9)}-host.example{i % 500}.com"
        events.append({
            "ts": ts, "src_ip": ip, "domain": domain,
            "qtype": rng.choice(["A", "A", "A", "TXT"]),
            "length": len(domain), "nxdomain": rng.random() < 0.05,
        })
    return events


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=50_000, help="number of synthetic events")
    ap.add_argument("--n-ips", type=int, default=2_000, help="distinct source IPs")
    ap.add_argument("--max-ips", type=int, default=500,
                     help="deliberately << n-ips, to force eviction and prove it bounds memory")
    ap.add_argument("--dga-batch-size", type=int, default=500,
                     help="also benchmark the batched DGA scoring path at this batch size -- "
                          "the throughput a real micro-batching collector would actually see")
    args = ap.parse_args()

    dga_model = joblib.load(MODEL_DIR / "dga_lightgbm.pkl")
    tunnel_model = joblib.load(MODEL_DIR / "tunnelling_isolation_forest.pkl")
    state_mgr = IPStateManager(max_ips=args.max_ips, ttl_seconds=10**9)  # LRU only, no TTL noise

    events = gen_events(args.n, args.n_ips)

    latencies_ms = []
    t_start = time.perf_counter()
    for event in events:
        t0 = time.perf_counter()
        process_event(event, dga_model, tunnel_model, state_mgr)
        latencies_ms.append((time.perf_counter() - t0) * 1000)
    elapsed = time.perf_counter() - t_start

    latencies_ms.sort()
    results = {
        "n_events": args.n,
        "n_distinct_ips": args.n_ips,
        "max_ips_configured": args.max_ips,
        "wall_seconds": elapsed,
        "events_per_second": args.n / elapsed,
        "latency_ms_mean": sum(latencies_ms) / len(latencies_ms),
        "latency_ms_p50": percentile(latencies_ms, 0.50),
        "latency_ms_p95": percentile(latencies_ms, 0.95),
        "latency_ms_p99": percentile(latencies_ms, 0.99),
        "active_ips_after_run": state_mgr.active_ips,
        "evicted_total": state_mgr.evicted_total,
    }

    print(f"{args.n:,} events / {args.n_ips:,} distinct IPs, max_ips={args.max_ips}")
    print(f"  {results['events_per_second']:,.0f} events/sec sustained "
          f"({elapsed:.2f}s wall time)")
    print(f"  latency (ms): mean={results['latency_ms_mean']:.4f} "
          f"p50={results['latency_ms_p50']:.4f} "
          f"p95={results['latency_ms_p95']:.4f} "
          f"p99={results['latency_ms_p99']:.4f}")
    print(f"  IPStateManager stayed bounded: active_ips={results['active_ips_after_run']} "
          f"(<= max_ips={args.max_ips}) despite {args.n_ips:,} distinct IPs seen, "
          f"evicted_total={results['evicted_total']:,}")

    # The micro-batching path: what a real DNS collector buffering ~100ms
    # worth of queries at a time would actually see. Compared against a
    # DGA-ONLY single-event baseline (not the mixed DGA+tunnelling number
    # above, which would be an apples-to-oranges comparison since that
    # baseline also pays for state_mgr bookkeeping the batched path skips).
    n_batches = args.n // args.dga_batch_size
    if n_batches:
        batches = [events[i*args.dga_batch_size:(i+1)*args.dga_batch_size] for i in range(n_batches)]
        batch_events = n_batches * args.dga_batch_size

        t0 = time.perf_counter()
        for event in events[:batch_events]:
            check_dga(dga_model, event)
        dga_single_elapsed = time.perf_counter() - t0
        dga_single_rate = batch_events / dga_single_elapsed

        t0 = time.perf_counter()
        for batch in batches:
            check_dga_batch(dga_model, batch)
        batch_elapsed = time.perf_counter() - t0
        batch_rate = batch_events / batch_elapsed

        results["dga_batch_size"] = args.dga_batch_size
        results["dga_only_single_event_per_second"] = dga_single_rate
        results["dga_batched_events_per_second"] = batch_rate
        print(f"\nDGA-only comparison (batch_size={args.dga_batch_size}, same {batch_events:,} events "
              f"both ways):")
        print(f"  single-event: {dga_single_rate:,.0f} events/sec")
        print(f"  batched:      {batch_rate:,.0f} events/sec  ({batch_rate/dga_single_rate:.1f}x)")

    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nSaved results to {OUT_PATH}")


if __name__ == "__main__":
    main()
