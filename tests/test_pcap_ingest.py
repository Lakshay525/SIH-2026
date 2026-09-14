"""
scripts/09_ingest_pcap.py is the real ingestion boundary this pipeline was
"architecturally ready for" but didn't have -- everything from
src/pipeline/engine.py onward only knows the JSONL event schema, and this is
what actually produces that schema from a real captured .pcap file instead
of a synthetic replay log.

Builds a small, well-formed synthetic pcap with scapy (crafting packets, not
sniffing -- no npcap/live-capture needed) covering the three real cases that
matter: a query with a matching NOERROR response, a query with a matching
NXDOMAIN response, and an orphan query with no response in the capture
(a truncated capture must not silently drop that traffic).
"""
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

pytest.importorskip("scapy.all", reason="scapy not installed")
from scapy.all import DNS, DNSQR, DNSRR, IP, UDP, wrpcap  # noqa: E402


def _load_ingest_module():
    """
    scripts/09_ingest_pcap.py can't be `import`ed normally -- a module name
    can't start with a digit. Load it directly from its file path instead of
    duplicating its logic here or renaming the script out of its established
    numbered-scripts convention.
    """
    path = REPO_ROOT / "scripts" / "09_ingest_pcap.py"
    spec = importlib.util.spec_from_file_location("ingest_pcap", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ingest_pcap = _load_ingest_module()


@pytest.fixture
def synthetic_pcap(tmp_path):
    pkts = []
    t = 1000.0

    q1 = IP(src="10.0.0.5", dst="8.8.8.8") / UDP(sport=53001, dport=53) / \
        DNS(id=1, qr=0, qd=DNSQR(qname="google.com", qtype="A"))
    q1.time = t
    r1 = IP(src="8.8.8.8", dst="10.0.0.5") / UDP(sport=53, dport=53001) / \
        DNS(id=1, qr=1, rcode=0, qd=DNSQR(qname="google.com", qtype="A"),
            an=DNSRR(rrname="google.com", rdata="1.2.3.4"))
    r1.time = t + 0.01

    q2 = IP(src="203.0.113.7", dst="8.8.8.8") / UDP(sport=53002, dport=53) / \
        DNS(id=2, qr=0, qd=DNSQR(qname="deadbeef01234.tunnel.example.net", qtype="TXT"))
    q2.time = t + 1.0
    r2 = IP(src="8.8.8.8", dst="203.0.113.7") / UDP(sport=53, dport=53002) / \
        DNS(id=2, qr=1, rcode=3, qd=DNSQR(qname="deadbeef01234.tunnel.example.net", qtype="TXT"))
    r2.time = t + 1.02

    q3 = IP(src="10.0.0.9", dst="8.8.8.8") / UDP(sport=53003, dport=53) / \
        DNS(id=3, qr=0, qd=DNSQR(qname="orphan-query.example.org", qtype="A"))
    q3.time = t + 2.0

    path = tmp_path / "test_capture.pcap"
    wrpcap(str(path), [q1, r1, q2, r2, q3])
    return path


def test_extract_events_matches_query_to_its_response(synthetic_pcap):
    from scapy.utils import rdpcap

    packets = rdpcap(str(synthetic_pcap))
    events = list(ingest_pcap.extract_events(packets))

    assert len(events) == 3
    by_ip = {e["src_ip"]: e for e in events}

    assert by_ip["10.0.0.5"]["domain"] == "google.com"
    assert by_ip["10.0.0.5"]["qtype"] == "A"
    assert by_ip["10.0.0.5"]["nxdomain"] is False

    assert by_ip["203.0.113.7"]["domain"] == "deadbeef01234.tunnel.example.net"
    assert by_ip["203.0.113.7"]["qtype"] == "TXT"
    assert by_ip["203.0.113.7"]["nxdomain"] is True   # matched a real NXDOMAIN response

    # Orphan query: no response in the capture -> nxdomain defaults to False
    # (unknown), the event is still emitted, not dropped.
    assert by_ip["10.0.0.9"]["domain"] == "orphan-query.example.org"
    assert by_ip["10.0.0.9"]["nxdomain"] is False


def test_extracted_events_are_valid_per_the_ingest_schema(synthetic_pcap):
    from scapy.utils import rdpcap
    from src.pipeline.event_schema import validate_event

    packets = rdpcap(str(synthetic_pcap))
    for event in ingest_pcap.extract_events(packets):
        validate_event(event)  # must not raise -- extract_events already validates internally
