"""
Popularity allowlist -- the missing reputation layer.

A purely lexical DGA classifier has no way to know that "wettringer-
modellbauforum.de" is a real German hobby forum and not algorithmic junk; it
only sees consonant runs and a bad n-gram score. That costs a ~5% false
positive rate on real traffic, which is the dominant source of noise in the
live replay.

The standard mitigation, used by essentially every production DGA detector, is
a popularity allowlist: a domain sitting in the global top-N most-queried
domains is, by construction, not a freshly-generated algorithmic C2 name. DGA
domains are registered (or NXDOMAIN) days-to-hours before use and never
accumulate global query volume, so the two populations barely overlap.

Loaded from the Cisco Umbrella top-1M list already vendored in data/ -- read
straight from the zip, rank-ordered, so "top N" is meaningful. (Note that
data/benign_domains.txt cannot be used for this: scripts/00 builds it through
a Python set(), which discards rank order entirely.)

This is a suppression layer, NOT a detection improvement -- it trades a small
amount of recall for a large amount of precision, and the trade is measured
and reported in README.md rather than asserted. It is off by default; callers
opt in explicitly.
"""
import zipfile
from pathlib import Path

from src.features.lexical import extract_registrable_domain

DEFAULT_TOP_N = 100_000


class PopularityAllowlist:
    """Registrable domains from the top-N of a rank-ordered popularity list."""

    def __init__(self, domains=None):
        self._domains = domains or set()

    def __len__(self):
        return len(self._domains)

    def __contains__(self, domain: str) -> bool:
        return extract_registrable_domain(domain) in self._domains

    @classmethod
    def from_umbrella_zip(cls, zip_path: Path, top_n: int = DEFAULT_TOP_N):
        """
        Umbrella's top-1m.csv is "rank,domain" with no header, already sorted
        by rank -- so reading the first top_n lines gives the most-queried
        domains globally.
        """
        zip_path = Path(zip_path)
        if not zip_path.exists():
            return cls(set())

        domains = set()
        with zipfile.ZipFile(zip_path) as z:
            with z.open(z.namelist()[0]) as f:
                for i, raw in enumerate(f):
                    if i >= top_n:
                        break
                    line = raw.decode("utf-8", errors="ignore").strip()
                    if not line:
                        continue
                    domains.add(extract_registrable_domain(line.split(",")[-1]))
        return cls(domains)
