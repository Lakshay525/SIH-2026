"""Standardized alert format."""
import time
import uuid

def make_alert(flow_id: str, threat_class: str, confidence: float, evidence: dict) -> dict:
    """
    threat_class: "DGA" or "DNS_TUNNELLING"
    confidence: 0.0 - 1.0
    evidence: e.g. {"domain": "xqzplvmno.com", "entropy": 4.2}
    """
    return {
        "alert_id": str(uuid.uuid4()),
        "timestamp": time.time(),
        "flow_id": flow_id,
        "threat_class": threat_class,
        "confidence": confidence,
        "evidence": evidence,
    }