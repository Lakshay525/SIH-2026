"""Standardized alert format."""
import time
import uuid


def make_alert(flow_id: str, threat_class: str, confidence: float, evidence: dict,
               observed_ts: float = None) -> dict:
    """
    threat_class: "DGA" or "DNS_TUNNELLING"
    confidence: 0.0 - 1.0
    evidence: e.g. {"domain": "xqzplvmno.com", "entropy": 4.2}
    observed_ts: the capture timestamp of the DNS event that triggered this
        alert. THIS is what goes in "timestamp" -- an alert must be stamped
        with when the traffic was observed, not when we happened to get around
        to processing it. Replaying the same captured stream twice has to
        produce byte-identical timestamps, or the "read-only ingest preserves a
        clean chain of custody" property is not actually true: a forensic
        timeline built from these alerts would otherwise show when the analyst
        ran the tool, not when the attack happened. Falls back to wall-clock
        only when a caller genuinely has no observed timestamp to give.
    """
    # Determine severity based on the confidence score
    if confidence >= 0.85:
        severity = "CRITICAL"
    elif confidence >= 0.60:
        severity = "HIGH"
    elif confidence >= 0.30:
        severity = "MEDIUM"
    else:
        severity = "LOW"

    return {
        "alert_id": str(uuid.uuid4()),
        "timestamp": observed_ts if observed_ts is not None else time.time(),
        "flow_id": flow_id,
        "threat_class": threat_class,
        "severity": severity,
        "confidence": confidence,
        "evidence": evidence,
    }
