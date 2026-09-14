"""
Protocol-realistic synthetic DNS traffic windows -- normal vs. three real DNS
tunnelling tools' behavioral signatures. Columns match
src/models/tunnelling_detector.FEATURE_COLS: [query_rate, unique_subdomains,
avg_query_len, txt_ratio, nxdomain_rate, non_a_ratio, record_type_entropy,
label], plus an informational "tool" column (not used by training) recording
which behaviour a "tunnel" row models.

This is still synthetic (no real pcap was captured), but every parameter below
is grounded in how these tools actually behave on the wire, not arbitrary
gaussian noise -- including non_a_ratio/record_type_entropy, whose values here
are DIRECTLY calibrated against real per-query qd_qtype data from the Mendeley
tunnel-tool dataset (see scripts/03b, which computes these two features from
real records rather than assuming them):

  - iodine (raw/UDP mode): maximizes throughput via very frequent small
    round-trips, base32-encodes payload into the query name (near the 63-byte
    label / 253-byte name ceiling to amortize overhead), and since the tunnel
    server is authoritative for the domain almost every query resolves
    (low NXDOMAIN). Real measurement: 100% NULL records, near-zero
    record-type entropy (one type, not a mix).
  - dnscat2: a chattier C2/exfil channel -- hex-encodes each chunk (less
    space-efficient than base32, so shorter names), polls more like a
    beacon than a throughput benchmark. The real Mendeley dataset's closest
    analog (dnscapy, a related but distinct tool) measured ~53%/47%
    TXT/CNAME -- i.e. genuinely near-maximum entropy for 2 record types
    (~1.0 bit) -- modeled here as the one archetype with real type mixing.
  - dns2tcp: tunnels a full TCP stream, so it behaves like a sustained,
    MTU-constrained pipe -- steady high query rate, moderate name lengths to
    avoid UDP fragmentation. Real measurement: 100% TXT, near-zero entropy.

Real-data finding that shaped this (see README "Two more bugs..."):
non_a_ratio turned out to separate real tunnel tools from real normal
traffic far better than txt_ratio alone (1.00 vs 0.00 for all four real
tools, vs. txt_ratio being 0.0 for real iodine/tuns since they don't use
TXT at all) -- record_type_entropy is real but narrower, since 3 of the 4
real tools commit to a single non-A record type per session and only show
up as "mixed" for a tool like dnscapy.

To swap in fully real traffic: parse real pcaps/logs into rows with the same
columns and save as parquet at config.TUNNEL_DATASET_PATH -- or see
scripts/03b_build_real_tunnel_holdout.py, which folds in a real, labeled
public dataset as a held-out *evaluation* set rather than training data.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import TUNNEL_DATASET_PATH

SEED = 42


def gen_normal(rng, n):
    """Realistic benign resolver traffic for one host over a 60s window."""
    query_rate = rng.poisson(9, n).clip(1, None)
    # Most hosts only ever touch a handful of distinct names per window
    # (repeat visits + resolver retries + a few CDN hostnames); a benign
    # heavy-tail (a streaming/retail client fanning out across many CDN
    # edge subdomains) is mixed in on purpose so "high unique_subdomains"
    # alone isn't a free giveaway for the model.
    heavy_tail = rng.random(n) < 0.05
    unique_subdomains = np.where(
        heavy_tail,
        rng.poisson(12, n),
        rng.poisson(2, n),
    )
    unique_subdomains = np.minimum(unique_subdomains, query_rate)
    # mean=25.6, std=15.6 measured directly off data/benign_domains.txt (real
    # Umbrella/Tranco FQDNs) -- real passively-observed names run much longer
    # and more spread out than a naive guess, mostly due to CDN/tracking
    # subdomains, and using a tight synthetic distribution here made ordinary
    # long-but-benign FQDNs look anomalous to the trained Isolation Forest.
    avg_query_len = rng.normal(26, 15, n).clip(4, None)
    txt_ratio = rng.beta(1, 30, n)       # SPF/DKIM lookups are a small minority
    nxdomain_rate = rng.beta(1, 20, n)   # typos / stale cache entries
    non_a_ratio = rng.beta(1, 25, n)     # real measurement: ~0.00
    record_type_entropy = rng.beta(1, 20, n) * 0.3  # almost always single-type (A)
    return pd.DataFrame({
        "query_rate": query_rate,
        "unique_subdomains": unique_subdomains,
        "avg_query_len": avg_query_len,
        "txt_ratio": txt_ratio,
        "nxdomain_rate": nxdomain_rate,
        "non_a_ratio": non_a_ratio,
        "record_type_entropy": record_type_entropy,
        "label": "normal",
        "tool": "normal",
    })


def gen_iodine(rng, n):
    query_rate = rng.normal(70, 30, n).clip(10, None)
    unique_subdomains = (query_rate * rng.uniform(0.85, 1.0, n)).round()
    avg_query_len = rng.normal(210, 25, n).clip(63, 253)  # near name-length ceiling
    txt_ratio = rng.beta(1, 15, n)          # real measurement: 0% TXT (it's NULL, not TXT)
    nxdomain_rate = rng.beta(1, 25, n)      # tunnel server is authoritative -> mostly NOERROR
    non_a_ratio = rng.beta(30, 1, n)        # real measurement: 100% non-A (NULL)
    record_type_entropy = rng.beta(1, 15, n) * 0.4  # real measurement: ~0.01 bits, single type
    return pd.DataFrame({
        "query_rate": query_rate, "unique_subdomains": unique_subdomains,
        "avg_query_len": avg_query_len, "txt_ratio": txt_ratio,
        "nxdomain_rate": nxdomain_rate, "non_a_ratio": non_a_ratio,
        "record_type_entropy": record_type_entropy,
        "label": "tunnel", "tool": "iodine",
    })


def gen_dnscat2(rng, n):
    query_rate = rng.normal(35, 15, n).clip(5, None)
    unique_subdomains = (query_rate * rng.uniform(0.7, 0.95, n)).round()
    avg_query_len = rng.normal(100, 25, n).clip(20, None)  # hex overhead, smaller chunks
    txt_ratio = rng.beta(4, 4, n)           # mixed CNAME/TXT -- roughly half TXT
    nxdomain_rate = rng.beta(1, 25, n)
    non_a_ratio = rng.beta(30, 1, n)        # 100% non-A, just split across types
    # This is the one archetype with genuine record-type mixing -- calibrated
    # to the real dnscapy measurement of ~1.0 bit (near-max entropy for 2 types).
    record_type_entropy = rng.beta(4, 2, n) * 1.3
    return pd.DataFrame({
        "query_rate": query_rate, "unique_subdomains": unique_subdomains,
        "avg_query_len": avg_query_len, "txt_ratio": txt_ratio,
        "nxdomain_rate": nxdomain_rate, "non_a_ratio": non_a_ratio,
        "record_type_entropy": record_type_entropy,
        "label": "tunnel", "tool": "dnscat2",
    })


def gen_dns2tcp(rng, n):
    query_rate = rng.normal(75, 20, n).clip(15, None)
    unique_subdomains = (query_rate * rng.uniform(0.8, 1.0, n)).round()
    avg_query_len = rng.normal(135, 20, n).clip(30, None)  # MTU-constrained, steady chunks
    txt_ratio = rng.beta(20, 1, n)          # real measurement: 100% TXT
    nxdomain_rate = rng.beta(1, 25, n)
    non_a_ratio = rng.beta(30, 1, n)        # real measurement: 100% non-A
    record_type_entropy = rng.beta(1, 15, n) * 0.3  # real measurement: ~0.0 bits, single type
    return pd.DataFrame({
        "query_rate": query_rate, "unique_subdomains": unique_subdomains,
        "avg_query_len": avg_query_len, "txt_ratio": txt_ratio,
        "nxdomain_rate": nxdomain_rate, "non_a_ratio": non_a_ratio,
        "record_type_entropy": record_type_entropy,
        "label": "tunnel", "tool": "dns2tcp",
    })


def main():
    rng = np.random.default_rng(SEED)

    normal = gen_normal(rng, 2500)
    iodine = gen_iodine(rng, 300)
    dnscat2 = gen_dnscat2(rng, 300)
    dns2tcp = gen_dns2tcp(rng, 300)

    df = pd.concat([normal, iodine, dnscat2, dns2tcp], ignore_index=True)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    TUNNEL_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(TUNNEL_DATASET_PATH)

    print(f"Wrote {len(df):,} windows to {TUNNEL_DATASET_PATH}")
    print(df.groupby(["label", "tool"]).size())


if __name__ == "__main__":
    main()
