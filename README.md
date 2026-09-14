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
  behaviour, from seven aggregate features.
- A bounded-memory streaming state manager (`IPStateManager` + `IPState`) that computes those
  window features live from an ordered event stream, with two independent memory bounds (see
  "Bounded state" below).
- A standardized alert schema with a confidence→severity mapping.
- A streaming replay CLI (`scripts/07_stream_replay.py`) and a Streamlit dashboard that both run
  through the exact same detection code path (`src/pipeline/engine.py`) — not two drifting copies.
- A throughput/latency benchmark with measured (not invented) numbers.
- A test that mechanically enforces the "never actively probe/resolve anything" constraint,
  instead of that being true by accident.

- An **offline PCAP ingestion adapter** (`scripts/09_ingest_pcap.py`) that turns a real captured
  `.pcap`/`.pcapng` file into the same event schema everything else consumes — real DNS
  transactions (query matched to its response by transaction ID), real NXDOMAIN from the actual
  response rcode, not synthetic.
- **CI** (`.github/workflows/tests.yml`) running the full test suite on every push, and a
  **Dockerfile**/`docker-compose.yml` so the dashboard runs identically regardless of the host's
  Python version.
- **An ingest validation boundary** (`src/pipeline/event_schema.py`) — a malformed event (missing
  field, wrong type, corrupt JSON) is rejected with a specific reason and skipped instead of
  crashing the whole replay.

**Not built** (belongs to the shared team pipeline, out of scope for this slice): a NetFlow/IPFIX
ingestion path, TLS/QUIC or port-scan features, live packet capture (the PCAP adapter above reads a
capture file already on disk — this environment has no npcap/root capture capability, and a real
deployment reading directly off a live socket would itself be an active network participant, a
different and heavier claim than "read a capture file"; see "Real capture" below), and the unified
six-detector dashboard. This repo consumes and emits data at the boundary those would plug into
(a JSONL event stream in, a standardized alert schema out).

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

### Suffix resolution: from a hand-maintained list to the real Public Suffix List
The first fix for problem 3 above was a hand-written list of ~90 country-style and dynamic-DNS
suffixes — it worked, but re-derived a small slice of something that already exists and is
maintained by other people: the real [Mozilla Public Suffix
List](https://publicsuffix.org/list/) (`data/public_suffix_list.dat`, MPL-2.0, vendored snapshot).
`src/features/public_suffix.py` implements the actual PSL algorithm — right-to-left longest-match,
wildcard rules (`*.ck`), and exception rules (`!www.ck`) — verified against **73/73 applicable
vectors in the PSL project's own official test suite**
(`publicsuffix/list/tests/test_psl.txt`), including IDN suffixes that require punycode
normalization (`公司.cn` ⇄ `xn--55qx5d.cn`, via Python's builtin `idna` codec).

`extract_label()`/`extract_registrable_domain()` now delegate to this parser instead of the hand
list. Checking every suffix the hand list enumerated against the real PSL found **13 genuine
gaps** — providers the crowdsourced list never listed or later removed (`co.cc`/`cz.cc` shut down
years ago) — kept as a small, explicitly-justified supplement
(`_EMPIRICALLY_VERIFIED_EXTRA_SUFFIXES` in `lexical.py`) layered on top of the real list rather
than trusting either source alone.

Deliberately a **vendored snapshot file**, not `tldextract`: **tldextract fetches this same list
over the network on first use**, which would violate the read-only/no-egress constraint this whole
system rests on — `tests/test_one_way_constraints.py` fails it (and now explicitly greps for
`import tldextract` and checks `public_suffix.py` itself never touches a socket). Offline by
construction beats convenient. The trade-off is the snapshot needs periodic manual re-vendoring —
`data/public_suffix_list.dat` carries its own `VERSION` header so staleness is at least visible.

### Reputation layer (`src/features/reputation.py`) — opt-in DGA false-positive suppression
A purely lexical classifier cannot know that `wettringer-modellbauforum.de` is a real German hobby
forum; it only sees consonant runs and a bad n-gram score. That's the source of the residual ~5.4%
false-positive rate. The standard production mitigation is a **popularity allowlist**: a domain in
the global top-N most-queried list is by construction not a freshly-registered algorithmic C2 name.

`PopularityAllowlist` reads the Cisco Umbrella top-1M already vendored in `data/` (rank-ordered,
straight from the zip — note `data/benign_domains.txt` is *unusable* for this, because
`scripts/00` builds it through a `set()` and discards rank order). Top 100k lines, resolved to
registrable domains via the real PSL above → **19,865 distinct registrable domains** (more than
the 15,658 the old hand-suffix-list version produced from the same 100k lines, since the real PSL
correctly resolves far more countries' suffix structures instead of just the ones the hand list
happened to enumerate).

Measured trade, not asserted:

| | measurement |
|---|---|
| Benign dataset domains covered (FP-suppression reach) | **58.96%** |
| Genuine DGA domains wrongly suppressed (recall cost) | **0 of 447,378 (0.0000%)** |

That recall cost was **1.61% (7,192 domains)** before the dynamic-DNS fix above — allowlisting
matched on `duckdns.org` and would have whitelisted every dynamic-DNS C2 wholesale, handing
attackers a one-line bypass. Matching on the true registrable identity
(`bf65a853.duckdns.org`, not `duckdns.org`) closes it. This is a suppression layer, **off by
default** and opt-in via `python scripts/07_stream_replay.py --allowlist`, so both numbers stay
visible.

### Tunnelling detector (`src/pipeline/state_manager.py` + `src/models/tunnelling_detector.py`) — 7 window features
`query_rate`, `unique_subdomains`, `avg_query_len`, `txt_ratio`, `nxdomain_rate`, `non_a_ratio`,
`record_type_entropy`, computed per source IP over a 12×5s = 60-second sliding window.

**Bug fixed:** `nxdomain_rate` was never actually computed — every call site hardcoded it to `0.0`
after the fact, silently discarding one of the five original trained features on every real/live
run. `IPState` now tracks it directly.

**`non_a_ratio` and `record_type_entropy` added, and calibrated against real data, not guessed.**
Scoring the real Mendeley tunnel-tool dataset by record type showed:

| tool | `txt_ratio` | `non_a_ratio` | `record_type_entropy` |
|---|---|---|---|
| normal | 0.00 | **0.00** | 0.00 |
| dns2tcp | 1.00 | **1.00** | 0.00 |
| dnscapy | 0.53 | **1.00** | 0.98 bits |
| iodine | **0.00** | **1.00** | 0.01 bits |
| tuns | **0.00** | **1.00** | 0.00 |

`non_a_ratio` is a clean 0/1 separator for every real tool — including iodine and tuns, which
`txt_ratio` alone completely misses (they're NULL/CNAME-heavy, not TXT). `record_type_entropy` is
the weaker of the two on real data: three of the four tools commit to a single non-A record type
per session (entropy ≈ 0, indistinguishable from normal traffic on this feature alone), so it's a
real but narrower signal, added because it directly explains dnscapy's genuinely mixed
TXT/CNAME pattern rather than because it dominates.

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
| Random-split (optimistic, comparison only) | recall=0.903 precision=0.937 auc=0.977 |
| **Average held-out-family recall** | **0.803** |

That number moved through four retrains as real bugs were found and fixed, each measured rather
than assumed:

| Fix | Held-out recall |
|---|---|
| Baseline (40-word fallback dict, first-label extraction) | 0.750 |
| + real wordlist, `extract_label()` = `parts[-2]` | 0.814 |
| + hand-maintained multi-part-TLD/dynamic-DNS suffix list | 0.829 |
| + real Public Suffix List parser (final) | **0.803** |

The 3rd step (hand list) improved the average while silently breaking 9.25% of DGA domains onto
constant provider labels (`duckdns`, `ddns`, …) — fixing that repaired the specific damage and
improved the average. The 4th step (swapping the hand list for the real, official PSL — see
"Suffix resolution" below) gives up a little of that average (0.829→0.803) in exchange for
correctness verified against all 73 applicable vectors in the PSL project's own test suite,
covering every country's suffix structure instead of the ~30 this repo's DGA data happened to
exercise — a small, honestly-reported regression on this specific metric for a real completeness
gain. Random-split recall dropped alongside both suffix fixes (0.920→0.907→0.903), the expected,
healthier direction: the model can no longer pattern-match a delegation-point string as a DGA
marker across tens of thousands of samples and has to classify the actual attacker-controlled
label instead.

Per-family recall varies enormously and **doesn't move uniformly with a fix** — reported honestly
rather than smoothed into the average:

| family | 40-word dict, first-label | + wordlist, `parts[-2]` | + hand suffix list | + real PSL |
|---|---|---|---|---|
| `qhost` | — | — | 0.089 | **0.378** |
| `recjs` | — | — | 0.152 | **0.068** (still worse than baseline) |
| `qsnatch` | — | — | 0.000 | 0.000 (unaffected throughout) |

`recjs` getting worse and staying worse across two further fixes is a real result, not an error —
a feature-extraction change that helps the *aggregate* can still hurt an individual family whose
domains happened to correlate with the old (wrong) signal, and there's no guarantee a later,
more-correct fix undoes that for every family. `qsnatch` at 0.000 throughout suggests its DGA is
either dictionary-word-based or otherwise outside what these 8 lexical features can see at all —
see the script's full per-family printout for the rest.

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

- **100% recall on every real tunnel tool** in the dataset (dns2tcp, dnscapy, iodine, tuns) — held
  after adding the two new features below, which were themselves calibrated against this same
  real data (see "Features engineered").
- **0% specificity on the real "regular" holdout** (0/416 passed, all flagged — worse than the
  9.6% seen with the original 5-feature model). Rather than leave the explanation as an untested
  story, `scripts/04c_diagnose_real_specificity.py` runs an ablation: hold the real data fixed and
  swap one feature at a time to its synthetic-normal counterpart.

  | ablation (one feature replaced, other six real) | false-positive rate |
  |---|---|
  | baseline (as built) | 100.0% |
  | `query_rate` → synthetic median | 95.9% |
  | `unique_subdomains` → synthetic median | **0.0%** |
  | `avg_query_len` → synthetic median | 100.0% |
  | `txt_ratio` → synthetic median | 42.8% |
  | `nxdomain_rate` → synthetic median | 44.5% |
  | `non_a_ratio` → synthetic median | 93.5% |
  | `record_type_entropy` → synthetic median | **0.0%** |
  | `unique_subdomains` → realistic revisit rate, everything else real | **0.0%** |

  Same root cause as before, now measured with more features, not a new problem: the eval
  harness's `unique_subdomains == query_rate` construction (no real host produces that; the
  "regular" class here is a flat domain list, not a session log) still single-handedly explains
  the result — neutralising it alone collapses the false-positive rate to zero, same as it did
  with 5 features. Adding two genuinely strong real-tool separators (`non_a_ratio` is 1.00 vs 0.00,
  a clean split) made the *tunnel* signal stronger without fixing the *sessionisation* artifact
  driving the *normal* side, so the honest specificity number this dataset can produce got worse,
  not better, even though the two new features are individually well-justified. This is reported
  as-is rather than tuned away.

  **The honest corollary, stated plainly: this does not show the detector has good real-world
  specificity — it shows the false-positive number this dataset produces does not measure it.**
  Real-world specificity remains unmeasured, and cannot be measured without real per-host session
  logs, which this dataset's flat domain list cannot provide. See "Real capture" and
  **Limitations**.
- **Historical finding this drove**: real iodine and tuns traffic in this dataset showed
  `txt_ratio = 0.0` (they use NULL/CNAME records, not TXT), directly contradicting the original
  synthetic generator's assumption of TXT-heavy iodine traffic — the original 5-feature schema
  only tracked a TXT/non-TXT split and was blind to this. `non_a_ratio` and `record_type_entropy`
  (above) were added specifically to close this gap, calibrated against the same real measurement.
  Full numbers in `data/real_tunnel_eval.json`.

## Real DNS packet ingestion, and what "real capture" would take from here

`scripts/09_ingest_pcap.py` reads an actual `.pcap`/`.pcapng` file (e.g. from `tcpdump -w`, Zeek,
or a mirror-port capture) and turns it into the same event schema every other script consumes —
built with `scapy`, matching each query to its response by `(transaction ID, client IP, client
port)` so `nxdomain` comes from the response's real `rcode`, not a guess; an orphan query with no
matching response in the capture (a truncated capture) is still emitted, not silently dropped.
Verified end-to-end against a synthetic-but-wire-correct pcap crafted with scapy itself
(`tests/test_pcap_ingest.py`): a NOERROR transaction, an NXDOMAIN transaction, and an orphan query
all resolve correctly, and the output validates against `src/pipeline/event_schema.py` and flows
straight through `scripts/07_stream_replay.py` unchanged.

This is *offline* file parsing, not live sniffing — this environment has neither npcap/root
capture privileges nor, more importantly, an actual network tap to point at. What real capture
would take from here, concretely, closing the "real per-host session log" gap the ablation above
identifies:

```bash
# On a machine you control, capture your own ordinary DNS traffic for an hour
# (Linux/Mac):
sudo tcpdump -i <iface> -w my_session.pcap 'udp port 53'
# Windows, with Npcap installed:
"C:\Program Files\Wireshark\dumpcap.exe" -i <iface> -w my_session.pcap -f "udp port 53"

# Convert it with the tooling built here, and score it exactly like scripts/04c does:
python scripts/09_ingest_pcap.py my_session.pcap --out data/my_real_session.jsonl
python scripts/07_stream_replay.py --input data/my_real_session.jsonl --max-ips 5
```

That would be a real per-host session — repeated visits, real query timing, real NXDOMAIN
behaviour — closing exactly the gap the ablation above diagnosed, instead of another synthetic
approximation of one.

## Measured throughput

The PS requires stating and demonstrating a tested traffic rate — `scripts/08_benchmark.py` times
every event individually through the real pipeline (`process_event`), never estimates:

| events/sec | mean latency | p95 latency | p99 latency |
|---|---|---|---|
| **8,128** | **0.123 ms** | 0.164 ms | 0.210 ms |

Measured over 50,000 synthetic events across 2,000 distinct source IPs, with `max_ips=500` (far
below the 2,000 IPs generated) — `active_ips` stayed capped at exactly 500 and `evicted_total`
reached 49,500, confirming the LRU eviction actually fired under load, not just in a unit test.
Reproduce with `python scripts/08_benchmark.py`; the dashboard reads this same file rather than
asserting a canned "throughput sustained" string.

**This is a ~19x improvement (332 → 6,355 events/sec) from one bottleneck fix, verified
bit-identical.** Profiling found the entire cost was the sklearn wrapper: `predict_proba` on a
fresh one-row pandas `DataFrame` built per event cost 4.303ms; calling the underlying booster
directly on a raw numpy row costs 0.396ms — **max absolute difference 0.0** across the check
(`predict_proba` is a thin wrapper over the booster's raw output for a binary objective, so this
is not an approximation). Both models were switched: `check_dga` calls
`dga_model.booster_.predict(row)` directly, and the tunnelling `IsolationForest` derives `predict`
from `decision_function`'s sign (verified 0 disagreements over 2,000 samples) instead of walking
the forest twice. `src/models/tunnelling_detector.py`'s `train`/`score` were also made
numpy-consistent end-to-end, which incidentally silenced a recurring sklearn "X does not have
valid feature names" warning that the mixed DataFrame/array usage was causing.

Even at 8,128 events/sec, stated honestly: that's one process on one machine, adequate for a
mirror-port enclave but not sized against a core peering link — see **Limitations**.

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

## Ingest validation boundary

A real passive capture will produce malformed records occasionally (a truncated capture, a
corrupted log line, an upstream parser's own bug) — before `src/pipeline/event_schema.py` existed,
a missing `"ts"` key or a non-numeric `"length"` raised a raw `KeyError`/`TypeError` deep inside
`state_manager.py` and took the *entire replay* down over one bad line. `validate_event()` now sits
between untrusted input and the detection pipeline in both `scripts/07_stream_replay.py` and
`dashboard.py`: a malformed line is rejected with a specific, logged reason (missing field, wrong
type, an over-length name — a real DNS name can't exceed 253 octets, so anything longer signals
something upstream is already broken) and skipped, while everything else keeps flowing.
`tests/test_event_schema.py` includes the actual regression — three malformed lines mixed into a
stream no longer stop the other two valid ones from processing.

## How to run

### Option A: Docker (recommended for a judge/first-time run)
```bash
docker compose up --build
```
Opens the dashboard at `http://localhost:8501` once the healthcheck passes, without depending on
whatever Python version the host has. `data/` is bind-mounted, so `alerts.jsonl`/
`benchmark_results.json`/any freshly-generated demo stream persist on the host and are inspectable
without `docker cp`.

### Option B: local Python
```bash
pip install -r requirements.txt   # every dependency is version-pinned -- see below

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

# Real packet capture (see "Real DNS packet ingestion" above)
python scripts/09_ingest_pcap.py my_session.pcap --out data/my_real_session.jsonl
python scripts/07_stream_replay.py --input data/my_real_session.jsonl

# Throughput proof
python scripts/08_benchmark.py

# Dashboard
streamlit run dashboard.py

# Constraint tests
pytest tests/
```

### Reproducibility
- **`requirements.txt` is fully version-pinned** (`pandas==3.0.5`, not `pandas`) — every number in
  this README was measured against exactly these versions; an unpinned install on a different
  machine could silently reproduce different model behaviour.
- **CI** (`.github/workflows/tests.yml`) runs the full test suite, including the one-way
  constraint tests, on every push and pull request against `main` — not just claimed to pass
  locally.

## Limitations (stated honestly, not glossed over)

- **The suffix parser is the real PSL, but the 13-entry supplement is still hand-maintained.**
  `src/features/public_suffix.py` is verified against 73/73 official test vectors and needs no
  ongoing maintenance itself — but `_EMPIRICALLY_VERIFIED_EXTRA_SUFFIXES` in `lexical.py` is a
  small list of dynamic-DNS providers proven necessary by this specific dataset, and a future
  dataset could introduce a provider neither source lists. `tests/test_public_suffix.py` at least
  pins today's known gaps so a future PSL update that fills one is caught (the test fails loudly,
  on purpose, telling you to remove the now-redundant entry).
- **Real-world specificity of the tunnelling detector is unmeasured** (not "measured and bad", and
  not "fine"). The 0% figure from the public holdout is an artifact of that dataset's
  sessionisation, as the ablation in `scripts/04c` shows — but no dataset available here contains
  real per-host DNS session logs, so the number that would actually matter in deployment has never
  been measured. "Real DNS packet ingestion" above gives the exact recipe and the tooling
  (`scripts/09_ingest_pcap.py`) to close this on a machine with real capture capability — not
  done here because this environment has none.
- **The tunnelling training data is still synthetic**, just protocol-realistic and (for
  `non_a_ratio`/`record_type_entropy`) directly calibrated against real per-tool measurements
  rather than gaussian. The real-data holdout validates *recall* against actual tunnel-tool traffic
  (100% on all four tools), but no synthetic distribution is a substitute for a real capture.
- **No real NXDOMAIN signal in the real-data holdout** — the public dataset has no rcode field
  usable for it (see `scripts/03b`'s docstring), so `nxdomain_rate` is held at 0.0 for all
  real-derived sessions and isn't actually exercised by that particular evaluation. (The offline
  PCAP adapter above *does* derive real nxdomain from a real capture's response rcode — this gap is
  specific to the public Mendeley dataset, not to the pipeline generally.)
- **Held-out-family DGA recall varies enormously by family** — some families (dictionary-word-based
  DGAs like `suppobox`) are much harder for a purely lexical model than character-soup families.
  This is reported per-family by `scripts/02_train_dga_classifier.py`, not averaged away.
- **This is one of six required threat detectors** — it has no view of DDoS, botnet C2, encrypted
  malware, port scanning, or exfiltration traffic, and doesn't attempt to fuse signal across them.
- **The DGA classifier's residual false-positive rate is real and visible in the demo replay**,
  not hidden: replaying `scripts/06`'s real benign domain lookups raises false DGA alerts on
  legitimate foreign-language/unusual domains (German, Indonesian, Estonian sites). The popularity
  allowlist suppresses the majority of these at zero measured recall cost, but it is off by default
  and covers ~59% of benign domains — the uncovered tail still false-positives. This is why alerts
  carry a confidence score and supporting evidence for analyst triage rather than being wired to
  auto-block.
- **Throughput is adequate for a mirror-port enclave, not for a core peering link.** See "Measured
  throughput": a single process sustains ~8,128 events/sec after the numpy/booster fix. A busy
  gateway can produce far more DNS QPS than that, so real deployment would need either
  micro-batching (measured separately at ~0.0043ms/row amortised for a 300-row
  `booster_.predict` call — ~92x the current single-event path) or horizontal sharding by source
  IP. Neither is implemented as a production path here — stated as a limit, not hand-waved as
  "scales".
- **The PCAP adapter reads a capture file already on disk, not a live interface** — see "Scope:
  what's built vs. not built" for why (no npcap/root here, and a live-socket reader would itself be
  an active network participant, a different claim than "parse a passive capture").
- **Docker build is config-validated but not build-verified in this environment** — `docker compose
  config` confirms the compose file resolves correctly, but no Docker daemon was available to
  actually run `docker compose up` here. Stated plainly rather than claimed as tested.

## Project layout

```
config.py                              central paths
dashboard.py                           Streamlit UI, replays demo_stream.jsonl live
requirements.txt                       fully version-pinned
Dockerfile / docker-compose.yml        one-command reproducible run
.github/workflows/tests.yml            CI: full test suite on every push
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
  09_ingest_pcap.py                    real .pcap -> the same event schema, via scapy
src/
  features/lexical.py                  8 lexical features + extract_label() + suffix handling
  features/public_suffix.py            real Public Suffix List parser (offline, punycode-aware)
  features/reputation.py               popularity allowlist (opt-in FP suppression)
  models/dga_lightgbm.py               LightGBM wrapper + held_out_family_eval
  models/tunnelling_detector.py        Isolation Forest wrapper (numpy-consistent)
  pipeline/state_manager.py            IPState (per-IP) + IPStateManager (LRU/TTL across IPs)
  pipeline/engine.py                   the one shared process_event() path + AlertDeduper
  pipeline/alert_schema.py             make_alert() + severity mapping
  pipeline/event_schema.py             ingest validation boundary -- reject, don't crash
tests/
  test_one_way_constraints.py          mechanically enforces "never probes/resolves"
  test_alert_timestamps.py             alerts carry capture time; replays are reproducible
  test_sliding_window.py               all seven features decay; bounded-memory promotion
  test_label_extraction.py             CDN / multi-part TLD / dynamic-DNS label resolution
  test_public_suffix.py                real PSL parser vs. its own official test vectors
  test_event_schema.py                 malformed input is rejected, not a crash
  test_pcap_ingest.py                  real pcap -> event schema, via a scapy-crafted capture
```
