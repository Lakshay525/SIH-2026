"""
Builds data/demo_stream.jsonl -- a timestamp-ordered, multi-source-IP event
log for scripts/07_stream_replay.py and dashboard.py to replay. Unlike the
old scripts/05_live_demo.py (three hardcoded domains + one hardcoded IP),
this exercises the real shape of the problem:

  - Real background benign traffic from MANY source IPs (sampled from the
    real Tranco/Umbrella benign_domains.txt already in this repo) -- enough
    distinct IPs to actually force IPStateManager's LRU eviction, not just
    claim it's possible.
  - One DGA-infected host querying real DGArchive domain strings (not
    made-up gibberish) mixed into otherwise-normal lookups.
  - One DNS-tunnelling host emitting base32/hex-encoded subdomain chunks at
    a rate and record type consistent with the tools modeled in
    scripts/03_generate_tunnelling_dataset.py.

Each JSONL line is one observed DNS query -- exactly what a passive
mirror/tap can see (see src/pipeline/engine.py's `event` contract):
{ts, src_ip, domain, qtype, length, nxdomain}
"""
import base64
import csv
import json
import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, DGA_RAW_DIR

OUT_PATH = DATA_DIR / "demo_stream.jsonl"
BENIGN_FILE = DATA_DIR / "benign_domains.txt"
DGA_FAMILY_FILE = DGA_RAW_DIR / "emotet_dga.csv"

N_BENIGN_IPS = 300          # deliberately > the demo's max_ips, to force eviction
QUERIES_PER_BENIGN_IP = (3, 8)
SESSION_SECONDS = 45        # a burst of real activity, not spread over the whole capture
STREAM_SECONDS = 600        # 10-minute simulated capture window
SEED = 7


def load_benign_domains(n):
    with open(BENIGN_FILE, encoding="utf-8") as f:
        pool = [line.strip() for line in f if line.strip()]
    return random.sample(pool, min(n, len(pool)))


def load_real_dga_domains(n):
    if not DGA_FAMILY_FILE.exists():
        return [f"{''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=16))}.net" for _ in range(n)]
    with open(DGA_FAMILY_FILE, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    domains = [r["domain"].strip('"') for r in rows]
    return random.sample(domains, min(n, len(domains)))


def tunnel_chunk_domain(parent="tunnel-c2.example.net"):
    """iodine-style: base32-encode a random payload chunk into the label."""
    payload = random.randbytes(24)
    label = base64.b32encode(payload).decode().rstrip("=").lower()
    return f"{label}.{parent}"


def main():
    random.seed(SEED)
    events = []

    # 1. Background benign traffic across many source IPs.
    benign_pool = load_benign_domains(2000)
    for i in range(N_BENIGN_IPS):
        ip = f"10.{random.randint(0,254)}.{random.randint(0,254)}.{random.randint(1,254)}"
        n_q = random.randint(*QUERIES_PER_BENIGN_IP)
        # real hosts revisit a small personal set of sites, not a fresh one each time
        my_sites = random.sample(benign_pool, k=min(3, len(benign_pool)))
        # A real host isn't uniformly active for the whole 10-minute capture --
        # it has a burst of activity (an app opens, a page loads) then goes
        # quiet. Spreading queries uniformly over the full window instead
        # diluted every 60s window to near-zero query_rate, which looked just
        # as anomalous to the Isolation Forest as too much traffic.
        session_start = random.uniform(0, STREAM_SECONDS - SESSION_SECONDS)
        for _ in range(n_q):
            ts = session_start + random.uniform(0, SESSION_SECONDS)
            domain = random.choice(my_sites)
            events.append({
                "ts": ts, "src_ip": ip, "domain": domain, "qtype": "A",
                "length": len(domain), "nxdomain": random.random() < 0.03,
            })

    # 2. One DGA-infected host, real DGArchive (Emotet) domains mixed with
    #    a couple of normal lookups so it isn't 100% obviously malicious.
    dga_ip = "172.16.5.44"
    dga_domains = load_real_dga_domains(6)
    for domain in dga_domains:
        ts = random.uniform(0, STREAM_SECONDS)
        events.append({
            "ts": ts, "src_ip": dga_ip, "domain": domain, "qtype": "A",
            "length": len(domain), "nxdomain": random.random() < 0.4,  # C2 domains often not live yet
        })
    for domain in random.sample(benign_pool, 3):
        ts = random.uniform(0, STREAM_SECONDS)
        events.append({
            "ts": ts, "src_ip": dga_ip, "domain": domain, "qtype": "A",
            "length": len(domain), "nxdomain": False,
        })

    # 3. One DNS-tunnelling host: a burst of unique base32 chunks inside one
    #    60s window, mostly TXT, matching scripts/03's iodine-style profile.
    tunnel_ip = "203.0.113.77"
    burst_start = STREAM_SECONDS * 0.6
    for i in range(60):
        ts = burst_start + i * 0.7  # ~1.4 queries/sec, all inside one 60s window
        domain = tunnel_chunk_domain()
        events.append({
            "ts": ts, "src_ip": tunnel_ip, "domain": domain, "qtype": "TXT",
            "length": len(domain), "nxdomain": random.random() < 0.02,
        })

    events.sort(key=lambda e: e["ts"])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")

    print(f"Wrote {len(events):,} events across {N_BENIGN_IPS + 2} source IPs "
          f"over a {STREAM_SECONDS}s window to {OUT_PATH}")
    print(f"  DGA-infected host: {dga_ip} ({len(dga_domains)} real Emotet domains)")
    print(f"  Tunnelling host:   {tunnel_ip} (60 base32 TXT chunks in one window)")


if __name__ == "__main__":
    main()
