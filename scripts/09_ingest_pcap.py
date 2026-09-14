"""
Offline PCAP -> event JSONL adapter -- the ingestion boundary this pipeline
was "architecturally ready for" but didn't actually have. Everything from
src/pipeline/engine.py onward is ingestion-agnostic (it only knows the
{ts, src_ip, domain, qtype, length, nxdomain} event schema in
src/pipeline/event_schema.py); this script is what feeds a REAL passive
capture into that schema instead of a synthetic replay file.

Reads a .pcap/.pcapng file already saved to disk (e.g. from a mirror port or
`tcpdump -w capture.pcap`), NOT a live interface -- this repo has no npcap/
root capture capability, and more importantly, a real deployment ingesting
directly from a live socket/interface would itself be an active network
participant, which is a different (and heavier) claim than "read a capture
file someone else produced passively". `scripts/07_stream_replay.py` accepts
`-` for stdin, so this can still be pipelined into a live-ish workflow
(`tcpdump -w - 'udp port 53' | python scripts/09_ingest_pcap.py -
| python scripts/07_stream_replay.py --input -`) without this script itself
touching a socket.

One event per DNS TRANSACTION (matched by (transaction ID, client IP, client
port) between a query and its response), not one per packet -- this is what
lets nxdomain be a real signal instead of always False:
  - If a response for a query is found in the same capture, its rcode
    determines nxdomain (rcode 3 == NXDOMAIN) and the event's length is the
    query name's length.
  - If no matching response exists (truncated capture, an in-flight query at
    capture end), the event is still emitted from the query alone with
    nxdomain=False (unknown, not asserted) -- a truncated capture must not
    silently drop traffic.

    python scripts/09_ingest_pcap.py capture.pcap --out data/pcap_events.jsonl
    python scripts/09_ingest_pcap.py capture.pcap --out -   # stdout, for piping
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.pipeline.event_schema import validate_event, InvalidEvent

NXDOMAIN_RCODE = 3
# Imported lazily (scapy's import is slow and only scripts/tests that touch
# actual pcap data need it) but at MODULE level, not inside main() -- a
# previous version stashed this in a global only main() set, so calling
# extract_events() directly (as tests do) raised NameError.
from scapy.layers.dns import dnstypes as DNS_QTYPE_NAMES


def _qname_str(qname) -> str:
    if isinstance(qname, bytes):
        qname = qname.decode("utf-8", errors="replace")
    return qname.rstrip(".")


def extract_events(packets):
    """
    Yields validated events from an iterable of scapy packets. Two-pass:
    first index every DNS response by (id, client_ip, client_port), then walk
    queries in order and attach the matching response's outcome if any.
    """
    from scapy.layers.dns import DNS
    from scapy.layers.inet import IP, UDP

    responses = {}
    for pkt in packets:
        if DNS not in pkt or pkt[DNS].qr != 1 or IP not in pkt or UDP not in pkt:
            continue
        dns = pkt[DNS]
        # A response's UDP dport is the original client's query source port.
        key = (dns.id, pkt[IP].dst, pkt[UDP].dport)
        responses[key] = dns.rcode

    for pkt in packets:
        if DNS not in pkt or pkt[DNS].qr != 0 or pkt[DNS].qdcount == 0:
            continue
        if IP not in pkt or UDP not in pkt:
            continue
        dns = pkt[DNS]
        # dns.qd is a PacketListField (a real query can carry >1 question,
        # though virtually none do) -- take the first, per current scapy API.
        qd = dns.qd[0]
        domain = _qname_str(qd.qname)
        if not domain:
            continue
        qtype = DNS_QTYPE_NAMES.get(qd.qtype, str(qd.qtype))

        key = (dns.id, pkt[IP].src, pkt[UDP].sport)
        rcode = responses.get(key)
        nxdomain = (rcode == NXDOMAIN_RCODE) if rcode is not None else False

        raw = {
            "ts": float(pkt.time),
            "src_ip": pkt[IP].src,
            "domain": domain,
            "qtype": qtype,
            "length": len(domain),
            "nxdomain": nxdomain,
        }
        try:
            yield validate_event(raw)
        except InvalidEvent as e:
            print(f"[ingest] dropped one packet: {e}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pcap", help="path to a .pcap/.pcapng file")
    ap.add_argument("--out", default="-", help="output JSONL path, or '-' for stdout")
    args = ap.parse_args()

    from scapy.utils import rdpcap
    packets = rdpcap(args.pcap)
    out = sys.stdout if args.out == "-" else open(args.out, "w", encoding="utf-8")
    n = 0
    try:
        for event in extract_events(packets):
            out.write(json.dumps(event) + "\n")
            n += 1
    finally:
        if out is not sys.stdout:
            out.close()

    print(f"[ingest] {n} DNS query events extracted from {len(packets):,} packets "
          f"in {args.pcap}", file=sys.stderr)


if __name__ == "__main__":
    main()
