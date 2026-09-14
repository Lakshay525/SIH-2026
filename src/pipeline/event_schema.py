"""
Validates one observed-DNS-query event before it reaches the detection
pipeline. A real ingest boundary sits between untrusted captured data and the
rest of the system -- right now nothing does that here: a missing "ts" key,
a non-numeric "length", or a plain corrupt JSON line all crash
scripts/07_stream_replay.py mid-stream with a raw KeyError/TypeError deep
inside state_manager.py, taking the whole replay down over one bad line.

A passive capture WILL produce malformed records occasionally (truncated
captures, a parser upstream having its own bug, a corrupted log line) -- an
ingest boundary has to be built for that from day one, not added after the
first crash.
"""

REQUIRED_FIELDS = ("ts", "src_ip", "domain", "qtype", "length")


class InvalidEvent(ValueError):
    """Raised with a human-readable reason; callers decide skip vs. raise."""


def validate_event(raw: dict) -> dict:
    """
    Returns a cleaned event dict with the right types, or raises
    InvalidEvent with a specific, actionable reason. Never raises anything
    else (KeyError/TypeError) -- callers can catch just InvalidEvent.
    """
    if not isinstance(raw, dict):
        raise InvalidEvent(f"event is not a JSON object: {type(raw).__name__}")

    missing = [f for f in REQUIRED_FIELDS if f not in raw]
    if missing:
        raise InvalidEvent(f"missing required field(s): {', '.join(missing)}")

    ts = raw["ts"]
    if not isinstance(ts, (int, float)) or isinstance(ts, bool):
        raise InvalidEvent(f"ts must be a number, got {type(ts).__name__}")
    if ts < 0:
        raise InvalidEvent(f"ts must be non-negative, got {ts}")

    src_ip = raw["src_ip"]
    if not isinstance(src_ip, str) or not src_ip.strip():
        raise InvalidEvent(f"src_ip must be a non-empty string, got {src_ip!r}")

    domain = raw["domain"]
    if not isinstance(domain, str) or not domain.strip():
        raise InvalidEvent(f"domain must be a non-empty string, got {domain!r}")
    if len(domain) > 255:
        # A real DNS name can never exceed 253 octets on the wire -- this
        # size alone is a signal something upstream is already broken, not
        # just malicious. Reject rather than silently truncate/misfeature.
        raise InvalidEvent(f"domain exceeds max DNS name length: {len(domain)} chars")

    qtype = raw["qtype"]
    if not isinstance(qtype, str) or not qtype:
        raise InvalidEvent(f"qtype must be a non-empty string, got {qtype!r}")

    length = raw["length"]
    if not isinstance(length, (int, float)) or isinstance(length, bool) or length < 0:
        raise InvalidEvent(f"length must be a non-negative number, got {length!r}")

    nxdomain = raw.get("nxdomain", False)
    if not isinstance(nxdomain, bool):
        raise InvalidEvent(f"nxdomain must be a boolean if present, got {nxdomain!r}")

    return {
        "ts": float(ts),
        "src_ip": src_ip,
        "domain": domain,
        "qtype": qtype,
        "length": float(length),
        "nxdomain": nxdomain,
    }
