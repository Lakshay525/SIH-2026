# DGA Domains + DNS Tunnelling Detector

One slice of a six-detector passive-monitoring pipeline built for **SIH 2026, PS 26145**
("AI-Based Detection of Cyber Threats in Unidirectional IP Traffic"). This repo covers exactly
**one** of the six required threat classes:

> (c) DGA domains and DNS tunnelling: Entropy/n-gram analysis of DNS query names, plus
> query-length and record-type anomalies.

The other five (volumetric DDoS, botnet C2 beaconing, encrypted-malware/JA3-JA4, port scanning,
data exfiltration) are separate teammates' slices of the same wider project and are out of scope
here.

## Scope: what's built vs. not built

**Built:**
- A LightGBM classifier that scores individual DNS query names for DGA-likeness from lexical
  features alone (no network access, no reputation lookups).
- An Isolation Forest that scores 60-second, per-source-IP DNS traffic windows for tunnelling
  behaviour, from five aggregate features.
- A bounded-memory streaming state manager (`IPStateManager` + `IPState`) that computes those
  window features live from an ordered event stream, with two independent memory bounds (see
  "Bounded state" below).
- A standardized alert schema with a confidence→severity mapping.
- A streaming replay CLI (`scripts/07_stream_replay.py`) and a Streamlit dashboard that both run
  through the exact same detection code path (`src/pipeline/engine.py`) — not two drifting copies.
- A throughput/latency benchmark with measured (not invented) numbers.
- A test that mechanically enforces the "never actively probe/resolve anything" constraint,
  instead of that being true by accident.

**Not built** (belongs to the shared team pipeline, out of scope for this slice): the actual
PCAP/NetFlow ingestion layer, TLS/QUIC or port-scan features, and the unified six-detector
dashboard. This repo consumes and emits data at the boundary those would plug into (a JSONL event
stream in, a standardized alert schema out).

## Architecture

```
  DNS query event                 (passively observed: domain, qtype, length,
       |                           src_ip, nxdomain -- never resolved ourselves)
       v
  ┌─────────────────────┐   ┌──────────────────────────────┐
  │ lexical_features()  │   │ IPStateManager.add_event()    │
  │ (per-query, stateless)│  │ (per-source-IP sliding window)│
  └─────────┬────────────┘   └──────────────┬───────────────┘
            v                                v
     LightGBM classifier            Isolation Forest
     (DGA probability)              (tunnelling anomaly score)
            \                              /
             v                            v
            src/pipeline/engine.process_event()
                         |
                         v
              src/pipeline/alert_schema.make_alert()
              {alert_id, timestamp, flow_id, threat_class,
               severity, confidence, evidence}
                         |
              ┌──────────┴───────────┐
              v                      v
     data/alerts.jsonl        dashboard.py (Streamlit)
     (persisted, append-only)
```

## Features engineered

### DGA classifier (`src/features/lexical.py`) — 8 lexical features per query name
`length`, Shannon `entropy`, `digit_ratio`, `vowel_consonant_ratio`, `max_consonant_run`,
`ngram_score` (bigram log-likelihood against a real English wordlist), `dict_word_ratio` (fraction
of the name covered by real dictionary substrings), `hex_char_ratio`.

`ngram_score` and `dict_word_ratio` are the two features that actually generalize to *unseen* DGA
families — length/entropy alone separate obvious gibberish but don't transfer well (see the
held-out-family results below). Both depend on a real wordlist: `data/english_words.txt`
(`google-10000-english-no-swears.txt`, ~10k common English words, MIT-licensed, fetched from
`first20hours/google-10000-english`) — the code originally shipped with a 40-word placeholder
fallback that starved both features of signal.

**Label extraction fix:** the feature extractor originally took the *first* label of a domain
(`domain.split(".")[0]`). DGArchive training domains are always bare `random.tld` (2 labels), so
this was a no-op there — but real passively-observed FQDNs routinely carry a CDN/tracking
subdomain as the leftmost label (e.g. `65873f45162247cb.threatcast.guardsquare.com`), which reads
as high-entropy gibberish by pure chance and used to fire false CRITICAL alerts on ordinary
Twitch/Akamai/CDN traffic in the streaming demo. `extract_label()` now takes the label just before
the TLD instead — a heuristic (not full public-suffix-list-aware eTLD+1 extraction; multi-part
TLDs like `co.uk` are a known edge case).

### Tunnelling detector (`src/pipeline/state_manager.py` + `src/models/tunnelling_detector.py`) — 5 window features
`query_rate`, `unique_subdomains`, `avg_query_len`, `txt_ratio`, `nxdomain_rate`, computed per
source IP over a 12×5s = 60-second sliding window.

**Bug fixed:** `nxdomain_rate` was never actually computed — every call site hardcoded it to `0.0`
after the fact, silently discarding one of the five trained features on every real/live run.
`IPState` now tracks it directly.

## Bounded state (the actual "constant memory" story)

Two independent, testable bounds — not one claim standing in for both:

1. **`IPState` (per source IP):** a fixed 12-slot ring buffer. Each slot is stamped with the
   *global* bucket index that last wrote it and is reset the moment a write lands on a stale slot;
   `stats(current_bucket_idx)` also treats any slot that has scrolled out of the window as zero at
   read time. (The original version never reset a slot at all, so a "60-second sliding window" was
   actually a lifetime cumulative counter for any long-running IP — harmless for a one-shot demo
   burst, wrong for a real stream.) Unique-subdomain tracking promotes from an exact `set()` to a
   HyperLogLog sketch (`datasketch`, p=8) once an IP exceeds 40 uniques, bounding memory for one
   noisy IP too.
2. **`IPStateManager` (across many source IPs):** an `OrderedDict` LRU (`move_to_end` +
   `popitem(last=False)`, capped at `max_ips`) plus a TTL sweep (`expire(now_ts)`, driven by event
   timestamps, not wall-clock, so replays are reproducible) — bounds memory regardless of how many
   distinct source IPs a stream throws at it. This layer didn't exist before this pass at all;
   `IPState` alone only ever bounded memory for a single IP.

Proven, not just claimed, by `scripts/08_benchmark.py` (deliberately sets `max_ips` far below the
number of distinct IPs generated) and exercised in `tests/test_one_way_constraints.py`.

## Two more bugs found by actually running the streaming replay end-to-end

Building `scripts/06_generate_demo_stream.py` + `scripts/07_stream_replay.py` and looking at the
real alert output (not just unit tests) surfaced two more real problems no amount of offline
model evaluation would have caught:

- **Scoring every single event, including a window with only 1-2 queries so far, flagged huge
  numbers of ordinary IPs as tunnelling** — a thin/incomplete sample looks statistically unusual to
  an Isolation Forest purely from having too little data, independent of whether it looks like
  tunnelling. `engine.check_tunnelling` now waits for `MIN_QUERIES_FOR_TUNNEL_SCORING` (5) queries
  in the window before scoring it at all — real tunnelling sessions push 30-90+ queries into one
  window (see `scripts/03`), so genuine detection is barely delayed while the noise from
  brand-new/quiet IPs disappears.
- **An ongoing incident re-triggered a fresh, identical alert on every subsequent event** — a
  60-query tunnelling burst produced a dozen near-duplicate alerts instead of one. `AlertDeduper`
  (`src/pipeline/engine.py`) now suppresses repeat `(threat_class, flow_id)` alerts within a
  30-second cooldown, matching how a real alerting pipeline pages once per incident, not once per
  packet.

## Training & validation methodology

### DGA classifier
Two evaluations are reported, and only one of them should be trusted:
- **Random 80/20 split** — optimistic, kept only for comparison.
- **Held-out-family evaluation** (`held_out_family_eval`) — train on every DGA family except one,
  test only on the held-out family plus a benign sample. This is the honest generalization number,
  because a real DGA campaign is, by definition, a family the model has never seen.

| Metric | Value |
|---|---|
| Dataset | 137 real DGArchive families + real Tranco/Umbrella benign domains |
| Random-split (optimistic, comparison only) | recall=0.920 precision=0.946 auc=0.983 |
| **Average held-out-family recall** | **0.814** |

That 0.814 is *after* the wordlist + label-extraction fixes above — before them it was 0.750. Both
fixes measurably improved genuine cross-family generalization, not just cosmetic cleanup.

Per-family recall varies enormously (some families near-perfect, some — e.g. `suppobox`,
`qsnatch` — near zero). That spread is expected and reported honestly rather than averaged away;
see the script's full per-family printout.

### Tunnelling detector
Trained on a **protocol-realistic synthetic generator** (`scripts/03_generate_tunnelling_dataset.py`),
not gaussian noise — each "tunnel" row models a specific real tool's known wire behaviour (iodine,
dnscat2, dns2tcp: label-length ceiling, record-type bias, throughput profile). Isolation Forest is
fit on normal windows only (unsupervised), then scored against synthetic tunnel windows:

| | precision | recall | f1 |
|---|---|---|---|
| normal | 1.000 | 0.980 | 0.990 |
| tunnel | 0.947 | 1.000 | 0.973 |

**Then independently checked against a real, public, held-out dataset** (never trained on) —
"DNS Tunneling Queries for Binary/Multilabel Classification", Yakov Bubnov, Mendeley Data, DOI
`10.17632/mzn9hvdcxg.2`, CC BY 4.0 — grouped into synthetic sessions by
`scripts/03b_build_real_tunnel_holdout.py` and scored by `scripts/04b_evaluate_tunnel_on_real.py`:

- **100% recall on every real tunnel tool** in the dataset (dns2tcp, dnscapy, iodine, tuns).
- **~9.6% specificity on the real "regular" holdout** (40/416 correctly passed, 376 flagged) — a
  real, reported weakness, diagnosed rather than hidden: this dataset's "regular" class is a flat
  list of distinct root domains (like a top-domain list), not a real host's repeated-visit session
  log, so grouping it into fixed-size sessions manufactures artificially high `unique_subdomains`
  per session — a construction artifact this specific public dataset can't avoid, not necessarily
  evidence of the same false-positive rate against genuine single-host benign traffic. (This
  started at ~6% before `avg_query_len`'s synthetic "normal" distribution was recalibrated to the
  mean/stdev actually measured off `data/benign_domains.txt` — a real, if partial, improvement.)
  See **Limitations**.
- **Important, honest finding**: real iodine and tuns traffic in this dataset shows `txt_ratio =
  0.0` (they use NULL/CNAME records, not TXT) — directly contradicting the synthetic generator's
  assumption of TXT-heavy iodine traffic. The current 5-feature schema only tracks a TXT/non-TXT
  split, not full record-type diversity, so it's blind to this distinction — detection for these
  tools is being carried by `avg_query_len` alone, not `txt_ratio`. A real next iteration would add
  a record-type-entropy feature. Full numbers in `data/real_tunnel_eval.json`.

## Measured throughput

The PS requires stating and demonstrating a tested traffic rate — `scripts/08_benchmark.py` times
every event individually through the real pipeline (`process_event`), never estimates:

| events/sec | mean latency | p95 latency | p99 latency |
|---|---|---|---|
| 332 | 3.02 ms | 3.57 ms | 3.89 ms |

Measured over 50,000 synthetic events across 2,000 distinct source IPs, with `max_ips=500` (far
below the 2,000 IPs generated) — `active_ips` stayed capped at exactly 500 and `evicted_total`
reached 49,500, confirming the LRU eviction actually fired under load, not just in a unit test.
Reproduce with `python scripts/08_benchmark.py`; the dashboard reads this same file rather than
asserting a canned "throughput sustained" string.

Bottleneck, stated plainly: `check_dga`'s per-event pandas `DataFrame` construction + LightGBM
`predict_proba` call dominates cost (~2.7ms of the ~3ms) — `check_tunnelling` alone runs at
>100,000 events/sec. Batching multiple queries into one `predict_proba` call, or replacing the
per-call `DataFrame` wrapper with a raw numpy row, would be the next real optimization; not done
here so the benchmark reflects the code as actually shipped, not a hand-tuned hot path.

## Alert schema

```json
{
  "alert_id": "uuid4",
  "timestamp": 1234567890.12,
  "flow_id": "203.0.113.7 or the queried domain",
  "threat_class": "DGA | DNS_TUNNELLING",
  "severity": "LOW | MEDIUM | HIGH | CRITICAL",
  "confidence": 0.0,
  "evidence": { "...feature values that drove this alert..." }
}
```
`severity` is derived from `confidence` (≥0.85 CRITICAL, ≥0.60 HIGH, ≥0.30 MEDIUM, else LOW).
Alerts are appended to `data/alerts.jsonl` as they're produced — an append-only log, not a
transient in-process object, matching the "clean chain of custody" the passive-monitoring
constraint is meant to preserve.

## How to run

```bash
pip install -r requirements.txt

# Rebuild data + retrain both models from scratch (optional -- trained models are checked in)
python scripts/00_prepare_benign.py
python scripts/01_generate_dga_dataset.py
python scripts/02_train_dga_classifier.py
python scripts/03_generate_tunnelling_dataset.py
python scripts/04_train_tunnelling_detector.py
python scripts/03b_build_real_tunnel_holdout.py     # needs data/real_tunnel_domains.csv
python scripts/04b_evaluate_tunnel_on_real.py

# Streaming demo
python scripts/06_generate_demo_stream.py           # builds data/demo_stream.jsonl
python scripts/07_stream_replay.py                  # replays it, writes data/alerts.jsonl

# Throughput proof
python scripts/08_benchmark.py

# Dashboard
streamlit run dashboard.py

# Constraint tests
pytest tests/
```

## Limitations (stated honestly, not glossed over)

- **`extract_label()`'s TLD heuristic** is not public-suffix-list-aware — multi-part TLDs
  (`co.uk`, `com.au`, ...) will extract the wrong label. A real deployment should use `tldextract`
  or an equivalent PSL-based library instead.
- **`unique_subdomains` doesn't decay with the sliding window** the way the other four features
  do — it's a lifetime set/HyperLogLog for the life of the `IPState` object, not bucketed per
  60-second window. A very-long-running IP's unique count only ever grows.
- **The tunnelling training data is still synthetic**, just protocol-realistic rather than
  gaussian. The real-data holdout (above) validates recall against actual tunnel tool traffic, but
  the ~6% real-holdout specificity shows the synthetic "normal" distribution and this specific
  public dataset's "regular" class don't fully agree — and this dataset has no real per-host
  session logs to check against instead. A genuine per-host benign traffic capture (e.g. the
  sibling `data-exfiltration` branch's own Zeek lab approach) would be needed to close this
  honestly.
- **No real NXDOMAIN signal in the real-data holdout** — the public dataset has no rcode field
  usable for it (see `scripts/03b`'s docstring), so `nxdomain_rate` is held at 0.0 for all
  real-derived sessions and isn't actually exercised by that particular evaluation.
- **`txt_ratio` is a TXT/non-TXT binary**, not a full record-type distribution — real iodine/tuns
  traffic in the holdout dataset is NULL/CNAME-heavy, which this feature can't see (see above).
- **Held-out-family DGA recall varies enormously by family** — some families (dictionary-word-based
  DGAs like `suppobox`) are much harder for a purely lexical model than character-soup families.
  This is reported per-family by `scripts/02_train_dga_classifier.py`, not averaged away.
- **This is one of six required threat detectors** — it has no view of DDoS, botnet C2, encrypted
  malware, port scanning, or exfiltration traffic, and doesn't attempt to fuse signal across them.
- **The DGA classifier's ~5.4% false-positive rate (1 - 0.946 precision) is real and visible in the
  demo replay**, not hidden: replaying `scripts/06`'s ~900 real benign domain lookups through
  `scripts/07` raises roughly that many false DGA alerts on legitimate foreign-language/unusual
  domains (e.g. German, Indonesian, Estonian sites). A purely lexical model has no reputation or
  allowlist signal to fall back on — this is exactly why alerts carry a confidence score and
  evidence for analyst triage rather than being wired to auto-block.
- **Throughput is bottlenecked by per-event LightGBM inference** (~2.7ms of ~3ms per event) — see
  "Measured throughput" above for the concrete number and the un-taken optimization (batching /
  raw-numpy input) that would improve it.

## Project layout

```
config.py                              central paths
dashboard.py                           Streamlit UI, replays demo_stream.jsonl live
requirements.txt
data/                                  datasets, trained models, generated artifacts
scripts/
  00_prepare_benign.py                 merge Umbrella + Tranco -> benign_domains.txt
  01_generate_dga_dataset.py           DGArchive CSVs + benign -> dga_dataset.parquet
  02_train_dga_classifier.py           train + held-out-family eval -> dga_lightgbm.pkl
  03_generate_tunnelling_dataset.py    protocol-realistic synthetic -> tunnelling_windows.parquet
  03b_build_real_tunnel_holdout.py     real Mendeley dataset -> real_tunnel_eval_windows.parquet
  04_train_tunnelling_detector.py      train Isolation Forest -> tunnelling_isolation_forest.pkl
  04b_evaluate_tunnel_on_real.py       score the trained model against the real holdout
  05_live_demo.py                      minimal smoke test of both models
  06_generate_demo_stream.py           multi-IP timestamp-ordered event log -> demo_stream.jsonl
  07_stream_replay.py                  the actual streaming engine, JSONL in -> alerts.jsonl out
  08_benchmark.py                      measured throughput/latency + bounded-memory proof
src/
  features/lexical.py                  8 lexical features + extract_label()
  models/dga_lightgbm.py               LightGBM wrapper + held_out_family_eval
  models/tunnelling_detector.py        Isolation Forest wrapper
  pipeline/state_manager.py            IPState (per-IP) + IPStateManager (LRU/TTL across IPs)
  pipeline/engine.py                   the one shared process_event() path
  pipeline/alert_schema.py             make_alert() + severity mapping
tests/
  test_one_way_constraints.py          mechanically enforces "never probes/resolves"
```
