"""
Adaptive per-source-IP sliding-window state manager -- computes the tunnelling
detector's features live from a DNS event stream without unbounded memory growth.
Cheap set-based tracking by default; promotes to a HyperLogLog sketch only for
IPs that look suspicious (many unique subdomains).
"""
from datasketch import HyperLogLog

N_BUCKETS = 12          # 12 x 5-second buckets = 60-second sliding window
BUCKET_SECONDS = 5
PROMOTE_THRESHOLD = 40  # unique-subdomain count above which we switch to a sketch

class IPState:
    __slots__ = ("bucket_count", "bucket_sumlen", "bucket_txt",
                 "domain_set", "hll", "promoted")

    def __init__(self):
        self.bucket_count = [0] * N_BUCKETS
        self.bucket_sumlen = [0] * N_BUCKETS
        self.bucket_txt = [0] * N_BUCKETS
        self.domain_set = set()
        self.hll = None
        self.promoted = False

    def add(self, bucket_idx, domain, qtype, length):
        b = bucket_idx % N_BUCKETS
        self.bucket_count[b] += 1
        self.bucket_sumlen[b] += length
        if qtype == "TXT":
            self.bucket_txt[b] += 1

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

    def unique_estimate(self):
        return self.hll.count() if self.promoted else len(self.domain_set)

    def stats(self):
        count = sum(self.bucket_count)
        sum_len = sum(self.bucket_sumlen)
        txt = sum(self.bucket_txt)
        return {
            "query_rate": count,
            "avg_query_len": sum_len / count if count else 0,
            "txt_ratio": txt / count if count else 0,
            "unique_subdomains": self.unique_estimate(),
        }