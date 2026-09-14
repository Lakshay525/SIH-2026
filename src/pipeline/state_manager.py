"""
Adaptive per-source-IP sliding-window state manager -- computes the tunnelling
detector's features live from a DNS event stream without unbounded memory growth.

Two layers of bounding:
  1. IPState: a fixed-size ring buffer of N_BUCKETS x BUCKET_SECONDS per source IP.
     Each bucket slot is tagged with the *global* bucket index that last wrote to
     it and is reset the moment a write lands on a stale slot (and ignored at
     read time if a caller's current bucket has moved past it) -- so
     query_rate/avg_query_len/etc. actually decay once an IP goes quiet instead
     of accumulating forever. (The original version never reset a slot, so a
     "60-second sliding window" was actually a lifetime cumulative counter --
     harmless for a single one-shot demo burst, but wrong for a real stream.)
     Unique-subdomain tracking is bucketed the same way, so it decays with the
     window like every other feature (it used to be a single lifetime set that
     only ever grew, making one of the five features quietly non-windowed).
     Each bucket tracks its own uniques cheaply as a set and promotes to a
     HyperLogLog sketch if that single 5-second bucket blows past
     PROMOTE_THRESHOLD, bounding memory for one noisy IP too; stats() unions
     the live buckets to get the window's unique count.
  2. IPStateManager: bounds the *number of tracked IPs* -- IPState alone only
     bounds memory per IP, not across however many source IPs a stream throws
     at it. An OrderedDict LRU (move-to-end + popitem(last=False), capped at
     max_ips) plus a TTL sweep (expire(), driven by event timestamps rather
     than wall-clock so replays are reproducible) keep total memory bounded
     regardless of stream length or IP fan-out.
"""
import math
from collections import Counter, OrderedDict
from datasketch import HyperLogLog

N_BUCKETS = 12          # 12 x 5-second buckets = 60-second sliding window
BUCKET_SECONDS = 5
PROMOTE_THRESHOLD = 40  # uniques within ONE bucket above which we switch to a sketch
# HyperLogLog precision: 2^12 = 4096 registers (~4KB) per PROMOTED bucket.
# Measured error over the range a 5s bucket realistically sees: ~1.2% @100
# uniques, ~1.7% @500, ~0.8% @2000. p=8 was cheaper (256B) but hit 10.6%
# error at 2000 uniques and tripped datasketch's own accuracy warning.
# Worst-case memory is 12 buckets x 4KB = ~48KB for a single IP, and only for
# IPs actually flooding >PROMOTE_THRESHOLD distinct names per 5s bucket --
# see README "Bounded state" for the full max_ips x promoted-bucket math.
HLL_P = 12


def bucket_index(ts: float) -> int:
    """Convert an event timestamp into a monotonic global bucket index."""
    return int(ts // BUCKET_SECONDS)


class IPState:
    __slots__ = ("bucket_count", "bucket_sumlen", "bucket_qtypes", "bucket_nxdomain",
                 "bucket_epoch", "bucket_domains", "bucket_hll")

    def __init__(self):
        self.bucket_count = [0] * N_BUCKETS
        self.bucket_sumlen = [0] * N_BUCKETS
        self.bucket_qtypes = [Counter() for _ in range(N_BUCKETS)]
        self.bucket_nxdomain = [0] * N_BUCKETS
        self.bucket_epoch = [-1] * N_BUCKETS   # global bucket_idx last written to each slot
        self.bucket_domains = [set() for _ in range(N_BUCKETS)]
        self.bucket_hll = [None] * N_BUCKETS   # set once a bucket is promoted

    def _slot(self, bucket_idx):
        b = bucket_idx % N_BUCKETS
        if self.bucket_epoch[b] != bucket_idx:
            self.bucket_count[b] = 0
            self.bucket_sumlen[b] = 0
            self.bucket_qtypes[b] = Counter()
            self.bucket_nxdomain[b] = 0
            self.bucket_domains[b] = set()
            self.bucket_hll[b] = None
            self.bucket_epoch[b] = bucket_idx
        return b

    def add(self, bucket_idx, domain, qtype, length, nxdomain=False):
        b = self._slot(bucket_idx)
        self.bucket_count[b] += 1
        self.bucket_sumlen[b] += length
        self.bucket_qtypes[b][qtype] += 1
        if nxdomain:
            self.bucket_nxdomain[b] += 1

        if self.bucket_hll[b] is not None:
            self.bucket_hll[b].update(domain.encode("utf8"))
        else:
            self.bucket_domains[b].add(domain)
            if len(self.bucket_domains[b]) > PROMOTE_THRESHOLD:
                # This single 5s bucket is seeing a flood of distinct names --
                # swap it for a fixed-size sketch so memory stays bounded.
                hll = HyperLogLog(p=HLL_P)
                for d in self.bucket_domains[b]:
                    hll.update(d.encode("utf8"))
                self.bucket_domains[b] = set()
                self.bucket_hll[b] = hll

    def clear_bucket(self, bucket_idx):
        b = bucket_idx % N_BUCKETS
        self.bucket_count[b] = 0
        self.bucket_sumlen[b] = 0
        self.bucket_qtypes[b] = Counter()
        self.bucket_nxdomain[b] = 0
        self.bucket_domains[b] = set()
        self.bucket_hll[b] = None
        self.bucket_epoch[b] = bucket_idx

    def unique_estimate(self, live_buckets=None):
        """
        Distinct query names across the live buckets -- i.e. within the
        sliding window, not for all time. Exact while every live bucket is
        still a plain set (the overwhelmingly common case); approximate via a
        HyperLogLog union once any bucket has been promoted under load.
        """
        if live_buckets is None:
            live_buckets = range(N_BUCKETS)
        live_buckets = list(live_buckets)

        if not any(self.bucket_hll[b] is not None for b in live_buckets):
            union = set()
            for b in live_buckets:
                union |= self.bucket_domains[b]
            return len(union)

        merged = HyperLogLog(p=HLL_P)
        for b in live_buckets:
            if self.bucket_hll[b] is not None:
                merged.merge(self.bucket_hll[b])
            else:
                for d in self.bucket_domains[b]:
                    merged.update(d.encode("utf8"))
        return merged.count()

    def stats(self, current_bucket_idx=None):
        """
        Aggregate the last N_BUCKETS x BUCKET_SECONDS of traffic. Pass the
        caller's current bucket index to make this an honest *sliding* window:
        any slot whose last write has scrolled out of the window is treated as
        zero even if nothing has physically overwritten it yet. Omit it (the
        original behaviour) to just sum whatever the buffer currently holds.
        """
        if current_bucket_idx is None:
            live = list(range(N_BUCKETS))
        else:
            oldest_live_epoch = current_bucket_idx - N_BUCKETS + 1
            live = [b for b in range(N_BUCKETS) if self.bucket_epoch[b] >= oldest_live_epoch]

        count = sum(self.bucket_count[b] for b in live)
        sum_len = sum(self.bucket_sumlen[b] for b in live)
        nxdomain = sum(self.bucket_nxdomain[b] for b in live)

        qtypes = Counter()
        for b in live:
            qtypes.update(self.bucket_qtypes[b])
        txt = qtypes.get("TXT", 0)
        non_a = count - qtypes.get("A", 0)

        # Empirically grounded, not guessed: scoring the real Mendeley
        # tunnel-tool dataset (scripts/03b) showed non_a_ratio is 1.00 for
        # ALL FOUR real tunnelling tools vs 0.00 for real normal traffic --
        # a much stronger signal than txt_ratio alone, which is 0.0 for real
        # iodine/tuns traffic (they use NULL/CNAME, not TXT). record_type_
        # entropy is the weaker of the two on real data: 3 of 4 real tools
        # commit to a single non-A record type per session (entropy ~0, same
        # as normal traffic), so it mainly helps against tools that rotate
        # record types (only dnscapy did, at ~1.0 bit) -- kept because it's
        # a real, if narrower, signal, not because it dominates.
        entropy = 0.0
        if count:
            for n in qtypes.values():
                p = n / count
                if p:
                    entropy -= p * math.log2(p)

        return {
            "query_rate": count,
            "avg_query_len": sum_len / count if count else 0,
            "txt_ratio": txt / count if count else 0,
            "nxdomain_rate": nxdomain / count if count else 0,
            "non_a_ratio": non_a / count if count else 0,
            "record_type_entropy": entropy,
            "unique_subdomains": self.unique_estimate(live),
        }


class IPStateManager:
    """
    Bounds memory across *many* source IPs. Backs both halves of the "constant
    memory: HyperLogLog + LRU eviction" design: IPState above handles one noisy
    IP, this handles a stream that fans out across arbitrarily many IPs.
    """

    def __init__(self, max_ips: int = 100_000, ttl_seconds: float = 300.0):
        self.max_ips = max_ips
        self.ttl_seconds = ttl_seconds
        self._states = OrderedDict()  # ip -> [IPState, last_seen_ts]
        self.evicted_total = 0

    def __len__(self):
        return len(self._states)

    @property
    def active_ips(self) -> int:
        return len(self._states)

    def _get_or_create(self, ip: str) -> IPState:
        entry = self._states.get(ip)
        if entry is not None:
            self._states.move_to_end(ip)
            return entry[0]
        if len(self._states) >= self.max_ips:
            self._states.popitem(last=False)  # evict the least-recently-active IP
            self.evicted_total += 1
        state = IPState()
        self._states[ip] = [state, None]
        return state

    def add_event(self, ip: str, ts: float, domain: str, qtype: str, length: int,
                  nxdomain: bool = False):
        state = self._get_or_create(ip)
        state.add(bucket_index(ts), domain, qtype, length, nxdomain)
        self._states[ip][1] = ts
        self._states.move_to_end(ip)

    def get_stats(self, ip: str, current_ts: float = None) -> dict:
        entry = self._states.get(ip)
        if entry is None:
            return IPState().stats()
        current_bucket = bucket_index(current_ts) if current_ts is not None else None
        return entry[0].stats(current_bucket)

    def expire(self, now_ts: float) -> int:
        """Drop IPs idle longer than ttl_seconds. Returns the count evicted."""
        cutoff = now_ts - self.ttl_seconds
        evicted = 0
        # move_to_end on every touch keeps _states ordered least-recently-active
        # first, so we can stop at the first entry that's still fresh.
        while self._states:
            ip, (_, last_seen) = next(iter(self._states.items()))
            if last_seen is not None and last_seen < cutoff:
                self._states.popitem(last=False)
                evicted += 1
            else:
                break
        self.evicted_total += evicted
        return evicted
