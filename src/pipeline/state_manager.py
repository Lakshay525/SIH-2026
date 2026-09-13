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
     Cheap set-based unique-subdomain tracking promotes to a HyperLogLog sketch
     only for IPs that look suspicious (many unique subdomains), bounding
     memory for one noisy IP too.
  2. IPStateManager: bounds the *number of tracked IPs* -- IPState alone only
     bounds memory per IP, not across however many source IPs a stream throws
     at it. An OrderedDict LRU (move-to-end + popitem(last=False), capped at
     max_ips) plus a TTL sweep (expire(), driven by event timestamps rather
     than wall-clock so replays are reproducible) keep total memory bounded
     regardless of stream length or IP fan-out.
"""
from collections import OrderedDict
from datasketch import HyperLogLog

N_BUCKETS = 12          # 12 x 5-second buckets = 60-second sliding window
BUCKET_SECONDS = 5
PROMOTE_THRESHOLD = 40  # unique-subdomain count above which we switch to a sketch


def bucket_index(ts: float) -> int:
    """Convert an event timestamp into a monotonic global bucket index."""
    return int(ts // BUCKET_SECONDS)


class IPState:
    __slots__ = ("bucket_count", "bucket_sumlen", "bucket_txt", "bucket_nxdomain",
                 "bucket_epoch", "domain_set", "hll", "promoted")

    def __init__(self):
        self.bucket_count = [0] * N_BUCKETS
        self.bucket_sumlen = [0] * N_BUCKETS
        self.bucket_txt = [0] * N_BUCKETS
        self.bucket_nxdomain = [0] * N_BUCKETS
        self.bucket_epoch = [-1] * N_BUCKETS   # global bucket_idx last written to each slot
        self.domain_set = set()
        self.hll = None
        self.promoted = False

    def _slot(self, bucket_idx):
        b = bucket_idx % N_BUCKETS
        if self.bucket_epoch[b] != bucket_idx:
            self.bucket_count[b] = 0
            self.bucket_sumlen[b] = 0
            self.bucket_txt[b] = 0
            self.bucket_nxdomain[b] = 0
            self.bucket_epoch[b] = bucket_idx
        return b

    def add(self, bucket_idx, domain, qtype, length, nxdomain=False):
        b = self._slot(bucket_idx)
        self.bucket_count[b] += 1
        self.bucket_sumlen[b] += length
        if qtype == "TXT":
            self.bucket_txt[b] += 1
        if nxdomain:
            self.bucket_nxdomain[b] += 1

        if not self.promoted:
            self.domain_set.add(domain)
            if len(self.domain_set) > PROMOTE_THRESHOLD:
                self.hll = HyperLogLog(p=8)
                for d in self.domain_set:
                    self.hll.update(d.encode("utf8"))
                self.domain_set = None
                self.promoted = True
        else:
            self.hll.update(domain.encode("utf8"))

    def clear_bucket(self, bucket_idx):
        b = bucket_idx % N_BUCKETS
        self.bucket_count[b] = 0
        self.bucket_sumlen[b] = 0
        self.bucket_txt[b] = 0
        self.bucket_nxdomain[b] = 0
        self.bucket_epoch[b] = bucket_idx

    def unique_estimate(self):
        return self.hll.count() if self.promoted else len(self.domain_set)

    def stats(self, current_bucket_idx=None):
        """
        Aggregate the last N_BUCKETS x BUCKET_SECONDS of traffic. Pass the
        caller's current bucket index to make this an honest *sliding* window:
        any slot whose last write has scrolled out of the window is treated as
        zero even if nothing has physically overwritten it yet. Omit it (the
        original behaviour) to just sum whatever the buffer currently holds.
        """
        if current_bucket_idx is None:
            live = range(N_BUCKETS)
        else:
            oldest_live_epoch = current_bucket_idx - N_BUCKETS + 1
            live = [b for b in range(N_BUCKETS) if self.bucket_epoch[b] >= oldest_live_epoch]

        count = sum(self.bucket_count[b] for b in live)
        sum_len = sum(self.bucket_sumlen[b] for b in live)
        txt = sum(self.bucket_txt[b] for b in live)
        nxdomain = sum(self.bucket_nxdomain[b] for b in live)
        return {
            "query_rate": count,
            "avg_query_len": sum_len / count if count else 0,
            "txt_ratio": txt / count if count else 0,
            "nxdomain_rate": nxdomain / count if count else 0,
            "unique_subdomains": self.unique_estimate(),
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
