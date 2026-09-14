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

**Which label gets judged — three bugs, in sequence.** This one feature-extraction decision turned
out to be the single most consequential line in the DGA path, and it was wrong three different ways:

1. **Originally the *first* label** (`domain.split(".")[0]`). DGArchive training domains are bare
   `random.tld`, so this was a no-op in training — but real passively-observed FQDNs carry a
   CDN/tracking subdomain as the leftmost label (`65873f45162247cb.threatcast.guardsquare.com`),
   which reads as high-entropy gibberish by pure chance. It fired false CRITICAL alerts on ordinary
   Akamai/Twitch/CDN traffic.
2. **Fixing that with `parts[-2]` broke multi-part TLDs.** `infosys.co.in` was judged on the
   constant `"co"`. `.co.in`/`.ac.in`/`.net.in` are squarely in this project's deployment context,
   so this was a live failure, not a hypothetical.
3. **It also silently broke every dynamic-DNS-based DGA family** — and this was the expensive one.
   **9.25% of the DGA domains in this repo's own dataset (41,382 of 447,378)** are names like
   `bf65a853.duckdns.org`, where `parts[-2]` is the *provider* (`duckdns`), not the
   attacker-generated label. The classifier was being handed one of ~50 constant provider strings
   for 41k malicious samples. Affected families include `grandoreiro`, `g01`, `recjs`, `symmi`,
   `chaes`, `bamital`, `vidro`, `sutra` — and it lines up with the terrible per-family recalls
   (`recjs` 0.152, `qhost` 0.089) seen before the fix.

`extract_label()` now resolves the true registrable label against a bundled suffix list with two
parts: country-style two-label suffixes (`co.uk`, `co.in`, …) and **PSL "private section"
delegation points** — dynamic-DNS and free-hosting providers (`duckdns.org`, `ddns.net`,
`hopto.org`, `github.io`, `pages.dev`, …) where the public registers subdomains, so the
attacker-controlled label is one position further left. A generic-SLD-under-ccTLD heuristic
(`co.ls`, `com.cy`, `ac.at`) covers the long tail; it's gated on a two-letter TLD so it can't
misfire on real domains like `go.com`. After the fix, those 41,382 multi-label DGA domains resolve
to **37,224 distinct labels instead of ~50 provider constants**.

Deliberately a bundled static list rather than `tldextract`: **tldextract fetches the Public Suffix
List over the network on first use**, which would violate the read-only/no-egress constraint this
whole system rests on — `tests/test_one_way_constraints.py` fails it. Offline by construction beats
convenient. The trade-off is manual updating and less-than-full-PSL coverage.

### Reputation layer (`src/features/reputation.py`) — opt-in DGA false-positive suppression
A purely lexical classifier cannot know that `wettringer-modellbauforum.de` is a real German hobby
forum; it only sees consonant runs and a bad n-gram score. That's the source of the residual ~5.4%
false-positive rate. The standard production mitigation is a **popularity allowlist**: a domain in
the global top-N most-queried list is by construction not a freshly-registered algorithmic C2 name.

`PopularityAllowlist` reads the Cisco Umbrella top-1M already vendored in `data/` (rank-ordered,
straight from the zip — note `data/benign_domains.txt` is *unusable* for this, because
`scripts/00` builds it through a `set()` and discards rank order). Top 100k lines → **15,658
distinct registrable domains**.

Measured trade, not asserted:

| | measurement |
|---|---|
| Benign dataset domains covered (FP-suppression reach) | **62.1%** |
| Genuine DGA domains wrongly suppressed (recall cost) | **0 of 447,378 (0.0000%)** |

That recall cost was **1.61% (7,192 domains)** before the dynamic-DNS fix above — allowlisting
matched on `duckdns.org` and would have whitelisted every dynamic-DNS C2 wholesale, handing
attackers a one-line bypass. Matching on the true registrable identity
(`bf65a853.duckdns.org`, not `duckdns.org`) closes it. This is a suppression layer, **off by
default** and opt-in via `python scripts/07_stream_replay.py --allowlist`, so both numbers stay
visible.

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
   burst, wrong for a real stream.)

   **All five features now decay, including `unique_subdomains`.** That one used to be a single
   lifetime `set`/HyperLogLog that only ever grew, so even after the bucket-reset fix one of the
   five features was still not windowed — a long-running IP's unique count could never come back
   down, and the "60-second sliding window" claim was not fully true. Unique tracking is now
   bucketed like everything else: each bucket keeps its own exact `set`, promoting to a
   HyperLogLog only if that single 5-second bucket exceeds 40 uniques, and `stats()` unions the
   live buckets. Locked down by `tests/test_sliding_window.py`.

   Sketch sizing is measured, not guessed: `HLL_P=12` (4096 registers, ~4KB per *promoted* bucket)
   gives ~1.2% error at 100 uniques, ~1.7% at 500, ~0.8% at 2000. The previous `p=8` was cheaper
   (256B) but hit **10.6% error at 2000 uniques** and tripped datasketch's own accuracy warning.
   Worst case is 12 × 4KB ≈ 48KB for a single IP, and only for IPs actually flooding >40 distinct
   names per 5 seconds — i.e. the behaviour we're detecting. Note the honest corollary: a
   pathological population of `max_ips` simultaneously-flooding sources would be
   `max_ips × 48KB`, so `max_ips` is the knob that actually bounds worst-case memory.
2. **`IPStateManager` (across many source IPs):** an `OrderedDict` LRU (`move_to_end` +
   `popitem(last=False)`, capped at `max_ips`) plus a TTL sweep (`expire(now_ts)`, driven by event
   timestamps, not wall-clock, so replays are reproducible) — bounds memory regardless of how many
   distinct source IPs a stream throws at it. This layer didn't exist before this pass at all;
   `IPState` alone only ever bounded memory for a single IP.

Proven, not just claimed, by `scripts/08_benchmark.py` (deliberately sets `max_ips` far below the
number of distinct IPs generated) and exercised in `tests/test_one_way_constraints.py`.

## Alert timestamps: observed time, not processing time

`make_alert()` stamped every alert with `time.time()` — wall-clock *now* — and `engine.py` never
passed the event's own `ts` through. So every alert from a replay was stamped with whenever this
machine happened to process it. Replaying the same captured stream twice produced two different
timelines, which quietly turns "when did the attack happen" into "when did the analyst run the
tool" — directly contradicting the read-only/chain-of-custody property that is this system's whole
reason for existing.

`make_alert(..., observed_ts=event["ts"])` now carries capture time, falling back to wall-clock
only when a caller genuinely has no observed timestamp.
`tests/test_alert_timestamps.py` asserts the real property: **replaying the same capture twice
produces identical timestamps**, and a 2020-era capture is never stamped with today's date.

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
- **~9.6% specificity on the real "regular" holdout** (40/416 passed, 376 flagged). Rather than
  leave the explanation as an untested story, `scripts/04c_diagnose_real_specificity.py` runs an
  ablation: hold the real data fixed and swap one feature at a time to its synthetic-normal
  counterpart.

  | ablation (one feature replaced, other four real) | false-positive rate |
  |---|---|
  | baseline (as built) | 90.4% |
  | `query_rate` → synthetic median | **0.0%** |
  | `unique_subdomains` → synthetic median | **0.0%** |
  | `nxdomain_rate` → synthetic median | **0.0%** |
  | `avg_query_len` → synthetic median | 100.0% |
  | `txt_ratio` → synthetic median | 99.0% |
  | `unique_subdomains` → realistic revisit rate, everything else real | **0.0%** |

  The eval harness introduces **three** independent artifacts — fixed 12-query sessions, all-unique
  domains (`unique_subdomains == query_rate`, which no real host produces), and a constant
  `nxdomain_rate = 0.0` (the dataset has no rcode field, and an exact 0.0 is itself outside the
  trained distribution). **Neutralising any one of them alone collapses the false-positive rate to
  zero.** So the 9.6% figure is a property of how this public dataset has to be sessionised, not a
  measurement of the detector against real host traffic.

  **The honest corollary, stated plainly: this does not show the detector has good real-world
  specificity — it shows the 9.6% number does not measure it.** Real-world specificity remains
  unmeasured, and cannot be measured without real per-host session logs, which this dataset's flat
  domain list cannot provide. See **Limitations**.
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
python scripts/04c_diagnose_real_specificity.py     # ablation behind the real-holdout FP rate

# Streaming demo
python scripts/06_generate_demo_stream.py           # builds data/demo_stream.jsonl
python scripts/07_stream_replay.py                  # replays it, writes data/alerts.jsonl
python scripts/07_stream_replay.py --allowlist      # same, with popularity suppression on

# Throughput proof
python scripts/08_benchmark.py

# Dashboard
streamlit run dashboard.py

# Constraint tests
pytest tests/
```

## Limitations (stated honestly, not glossed over)

- **The bundled suffix list is not the full Public Suffix List.** It covers country-style
  two-label suffixes, the dynamic-DNS/free-hosting providers actually present in this dataset, and
  a generic-SLD-under-ccTLD heuristic for the long tail — but it is a static snapshot that needs
  manual updating, and a suffix outside it will still mis-resolve. This is a deliberate trade:
  `tldextract` would be more complete but fetches the PSL over the network, which the one-way
  constraint forbids. A vendored PSL *snapshot file* (parsed offline) would be the better long-term
  answer than a hand-maintained list.
- **Real-world specificity of the tunnelling detector is unmeasured** (not "measured and bad", and
  not "fine"). The 9.6% figure from the public holdout is an artifact of that dataset's
  sessionisation, as the ablation in `scripts/04c` shows — but no dataset available here contains
  real per-host DNS session logs, so the number that would actually matter in deployment has never
  been measured. Closing this needs a real capture (e.g. the sibling `data-exfiltration` branch's
  Zeek lab approach), not more analysis of this dataset.
- **The tunnelling training data is still synthetic**, just protocol-realistic rather than
  gaussian. The real-data holdout validates *recall* against actual tunnel-tool traffic (100% on
  all four tools), but no synthetic distribution is a substitute for a real capture.
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
- **The DGA classifier's residual false-positive rate is real and visible in the demo replay**,
  not hidden: replaying `scripts/06`'s real benign domain lookups raises false DGA alerts on
  legitimate foreign-language/unusual domains (German, Indonesian, Estonian sites). The popularity
  allowlist suppresses the majority of these at zero measured recall cost, but it is off by default
  and covers ~62% of benign domains — the uncovered tail still false-positives. This is why alerts
  carry a confidence score and supporting evidence for analyst triage rather than being wired to
  auto-block.
- **Throughput is adequate for a mirror-port enclave, not for a core peering link.** See "Measured
  throughput": a single process sustains low-thousands of events/sec after the numpy/booster fix.
  A busy gateway can produce far more DNS QPS than that, so real deployment would need the
  micro-batching path (measured at ~0.004ms/row amortised, ~100× the per-event path) or horizontal
  sharding by source IP. Neither is implemented — stated as a limit, not hand-waved as "scales".

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
  04c_diagnose_real_specificity.py     ablation: what actually drives the real-holdout FP rate
  05_live_demo.py                      minimal smoke test of both models
  06_generate_demo_stream.py           multi-IP timestamp-ordered event log -> demo_stream.jsonl
  07_stream_replay.py                  the actual streaming engine, JSONL in -> alerts.jsonl out
  08_benchmark.py                      measured throughput/latency + bounded-memory proof
src/
  features/lexical.py                  8 lexical features + extract_label() + suffix handling
  features/reputation.py               popularity allowlist (opt-in FP suppression)
  models/dga_lightgbm.py               LightGBM wrapper + held_out_family_eval
  models/tunnelling_detector.py        Isolation Forest wrapper (numpy-consistent)
  pipeline/state_manager.py            IPState (per-IP) + IPStateManager (LRU/TTL across IPs)
  pipeline/engine.py                   the one shared process_event() path + AlertDeduper
  pipeline/alert_schema.py             make_alert() + severity mapping
tests/
  test_one_way_constraints.py          mechanically enforces "never probes/resolves"
  test_alert_timestamps.py             alerts carry capture time; replays are reproducible
  test_sliding_window.py               all five features decay; bounded-memory promotion
  test_label_extraction.py             CDN / multi-part TLD / dynamic-DNS label resolution
```
