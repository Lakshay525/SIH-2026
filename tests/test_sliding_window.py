"""
The "60-second sliding window" has to actually slide -- for ALL five features.

Two real bugs are locked down here:
  1. Bucket counters were never reset, so query_rate/avg_query_len/txt_ratio/
     nxdomain_rate were lifetime cumulative counters wearing a window's name.
  2. unique_subdomains was a single lifetime set/HyperLogLog that only ever
     grew, so even after fix (1) one of the five features was still not
     windowed -- a long-running IP's unique count could never come back down.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.pipeline.state_manager import (
    IPState, IPStateManager, bucket_index, N_BUCKETS, PROMOTE_THRESHOLD,
)

WINDOW_BUCKETS = N_BUCKETS


def burst(state, n=60, start_ts=0.0, step=1.0, prefix="chunk"):
    for i in range(n):
        ts = start_ts + i * step
        state.add(bucket_index(ts), f"{i}-{prefix}.tunnel.example.com",
                  "TXT", 55, nxdomain=True)
    return bucket_index(start_ts + (n - 1) * step)


def test_all_five_features_decay_once_the_window_moves_on():
    state = IPState()
    last_bucket = burst(state)

    during = state.stats(last_bucket)
    assert during["query_rate"] == 60
    assert during["unique_subdomains"] == 60
    assert during["txt_ratio"] == 1.0
    assert during["nxdomain_rate"] == 1.0
    assert during["avg_query_len"] == 55.0

    after = state.stats(last_bucket + WINDOW_BUCKETS)
    for feature, value in after.items():
        assert value == 0, f"{feature} did not decay out of the window: {value}"


def test_unique_subdomains_is_exact_within_the_window():
    state = IPState()
    for i in range(10):
        state.add(bucket_index(i * 1.0), f"dup{i % 3}.example.com", "A", 20)
    assert state.stats(bucket_index(9.0))["unique_subdomains"] == 3


def test_unique_subdomains_counts_across_live_buckets_not_just_the_latest():
    """A name seen 40s ago is still inside a 60s window."""
    state = IPState()
    state.add(bucket_index(0.0), "early.example.com", "A", 20)
    state.add(bucket_index(40.0), "late.example.com", "A", 20)
    assert state.stats(bucket_index(40.0))["unique_subdomains"] == 2


def test_flooding_one_bucket_promotes_to_a_bounded_sketch():
    state = IPState()
    for i in range(500):
        state.add(bucket_index(0.0), f"{i}-flood.example.com", "TXT", 50)

    slot = bucket_index(0.0) % N_BUCKETS
    assert state.bucket_hll[slot] is not None, "bucket should have promoted"
    assert state.bucket_domains[slot] == set(), "promoted bucket must drop its exact set"

    estimate = state.stats(bucket_index(0.0))["unique_subdomains"]
    assert abs(estimate - 500) / 500 < 0.05, f"HLL estimate too far off: {estimate}"


def test_quiet_ip_does_not_accumulate_across_a_long_stream():
    """
    The regression that motivated all this: one query every 30s for an hour
    must never look like a high-rate window, no matter how long the stream runs.
    """
    mgr = IPStateManager(max_ips=10, ttl_seconds=10**9)
    ts = 0.0
    for _ in range(120):
        ts += 30.0
        mgr.add_event("10.0.0.1", ts, f"site{int(ts)}.example.com", "A", 20)

    stats = mgr.get_stats("10.0.0.1", current_ts=ts)
    assert stats["query_rate"] <= 3, f"quiet IP accumulated: {stats['query_rate']}"
    assert stats["unique_subdomains"] <= 3, (
        f"unique_subdomains accumulated over the whole stream: "
        f"{stats['unique_subdomains']}"
    )
