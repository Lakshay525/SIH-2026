"""
Protocol-realistic synthetic DNS traffic windows -- normal vs. three real DNS
tunnelling tools' behavioral signatures. Columns match
src/models/tunnelling_detector.FEATURE_COLS: [query_rate, unique_subdomains,
avg_query_len, txt_ratio, nxdomain_rate, label], plus an informational "tool"
column (not used by training) recording which behaviour a "tunnel" row models.

This is still synthetic (no real pcap was captured), but every parameter below
is grounded in how these tools actually behave on the wire, not arbitrary
gaussian noise:

  - iodine (raw/UDP mode): maximizes throughput via very frequent small
    round-trips, base32-encodes payload into the query name (near the 63-byte
    label / 253-byte name ceiling to amortize overhead), and since the tunnel
    server is authoritative for the domain almost every query resolves
    (low NXDOMAIN). Falls back to NULL/TXT-family records when raw mode is
    filtered.
  - dnscat2: a chattier C2/exfil channel -- hex-encodes each chunk (less
    space-efficient than base32, so shorter names), polls more like a
    beacon than a throughput benchmark, and commonly rides CNAME/TXT/MX
    records depending on config.
  - dns2tcp: tunnels a full TCP stream, so it behaves like a sustained,
    MTU-constrained pipe -- steady high query rate, moderate name lengths to
    avoid UDP fragmentation, and classically defaults to TXT records for its
    data channel.

Known limitation (documented in README.md too): the detector's feature set
only tracks a TXT/non-TXT split, not the full record-type distribution, so
NULL/CNAME/MX-heavy tunnel traffic is approximated here via an elevated
txt_ratio standing in for "non-standard record type usage" generally -- a
real capture would add a record-type-entropy feature instead.

To swap in fully real traffic: parse real pcaps/logs into rows with the same
six columns and save as parquet at config.TUNNEL_DATASET_PATH -- or see
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
    return pd.DataFrame({
        "query_rate": query_rate,
        "unique_subdomains": unique_subdomains,
        "avg_query_len": avg_query_len,
        "txt_ratio": txt_ratio,
        "nxdomain_rate": nxdomain_rate,
        "label": "normal",
        "tool": "normal",
    })


def gen_iodine(rng, n):
    query_rate = rng.normal(70, 30, n).clip(10, None)
    unique_subdomains = (query_rate * rng.uniform(0.85, 1.0, n)).round()
    avg_query_len = rng.normal(210, 25, n).clip(63, 253)  # near name-length ceiling
    txt_ratio = rng.beta(8, 3, n)          # NULL/TXT-family fallback, mostly non-A
    nxdomain_rate = rng.beta(1, 25, n)     # tunnel server is authoritative -> mostly NOERROR
    return pd.DataFrame({
        "query_rate": query_rate, "unique_subdomains": unique_subdomains,
        "avg_query_len": avg_query_len, "txt_ratio": txt_ratio,
        "nxdomain_rate": nxdomain_rate, "label": "tunnel", "tool": "iodine",
    })


def gen_dnscat2(rng, n):
    query_rate = rng.normal(35, 15, n).clip(5, None)
    unique_subdomains = (query_rate * rng.uniform(0.7, 0.95, n)).round()
    avg_query_len = rng.normal(100, 25, n).clip(20, None)  # hex overhead, smaller chunks
    txt_ratio = rng.beta(5, 4, n)          # mixed CNAME/TXT/MX
    nxdomain_rate = rng.beta(1, 25, n)
    return pd.DataFrame({
        "query_rate": query_rate, "unique_subdomains": unique_subdomains,
        "avg_query_len": avg_query_len, "txt_ratio": txt_ratio,
        "nxdomain_rate": nxdomain_rate, "label": "tunnel", "tool": "dnscat2",
    })


def gen_dns2tcp(rng, n):
    query_rate = rng.normal(75, 20, n).clip(15, None)
    unique_subdomains = (query_rate * rng.uniform(0.8, 1.0, n)).round()
    avg_query_len = rng.normal(135, 20, n).clip(30, None)  # MTU-constrained, steady chunks
    txt_ratio = rng.beta(10, 2, n)         # classically TXT-record data channel
    nxdomain_rate = rng.beta(1, 25, n)
    return pd.DataFrame({
        "query_rate": query_rate, "unique_subdomains": unique_subdomains,
        "avg_query_len": avg_query_len, "txt_ratio": txt_ratio,
        "nxdomain_rate": nxdomain_rate, "label": "tunnel", "tool": "dns2tcp",
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
