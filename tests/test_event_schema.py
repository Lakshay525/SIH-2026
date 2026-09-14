"""
The ingest boundary must reject malformed events instead of crashing deep
inside the pipeline. Before this validator existed, a missing "ts" key raised
a bare KeyError from inside state_manager.py, taking the whole replay down
over one bad line -- a passive capture WILL produce malformed records
occasionally (truncated captures, an upstream parser bug, log corruption).
"""
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.pipeline.event_schema import validate_event, InvalidEvent

VALID = {"ts": 12.5, "src_ip": "10.0.0.1", "domain": "example.com",
         "qtype": "A", "length": 11, "nxdomain": False}


def test_valid_event_passes_through():
    cleaned = validate_event(VALID)
    assert cleaned["ts"] == 12.5
    assert cleaned["domain"] == "example.com"
    assert cleaned["nxdomain"] is False


def test_nxdomain_defaults_to_false_when_absent():
    event = dict(VALID)
    del event["nxdomain"]
    assert validate_event(event)["nxdomain"] is False


@pytest.mark.parametrize("field", ["ts", "src_ip", "domain", "qtype", "length"])
def test_missing_required_field_is_rejected(field):
    event = dict(VALID)
    del event[field]
    with pytest.raises(InvalidEvent, match=field):
        validate_event(event)


@pytest.mark.parametrize("bad_ts", ["not-a-number", None, [1, 2], -5.0])
def test_bad_ts_is_rejected(bad_ts):
    event = dict(VALID, ts=bad_ts)
    with pytest.raises(InvalidEvent):
        validate_event(event)


@pytest.mark.parametrize("bad_domain", ["", "   ", 12345, None, "x" * 300])
def test_bad_domain_is_rejected(bad_domain):
    event = dict(VALID, domain=bad_domain)
    with pytest.raises(InvalidEvent):
        validate_event(event)


@pytest.mark.parametrize("bad_ip", ["", 12345, None])
def test_bad_src_ip_is_rejected(bad_ip):
    event = dict(VALID, src_ip=bad_ip)
    with pytest.raises(InvalidEvent):
        validate_event(event)


@pytest.mark.parametrize("bad_length", [-1, "twenty", None])
def test_bad_length_is_rejected(bad_length):
    event = dict(VALID, length=bad_length)
    with pytest.raises(InvalidEvent):
        validate_event(event)


def test_bad_nxdomain_type_is_rejected():
    with pytest.raises(InvalidEvent):
        validate_event(dict(VALID, nxdomain="yes"))


def test_non_dict_input_is_rejected():
    for bad in ["a string", 42, ["list", "not", "dict"], None]:
        with pytest.raises(InvalidEvent):
            validate_event(bad)


def test_a_malformed_line_does_not_stop_the_replay(tmp_path):
    """The actual regression: one bad line among good ones must not crash."""
    import json
    from src.pipeline.state_manager import IPStateManager
    from src.pipeline.engine import process_event
    import joblib
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from config import MODEL_DIR

    stream = tmp_path / "mixed.jsonl"
    lines = [
        json.dumps(VALID),
        "{not valid json",
        json.dumps({"ts": 1.0, "src_ip": "10.0.0.2"}),  # missing fields
        json.dumps(dict(VALID, ts=2.0)),
    ]
    stream.write_text("\n".join(lines), encoding="utf-8")

    dga_model = joblib.load(MODEL_DIR / "dga_lightgbm.pkl")
    tunnel_model = joblib.load(MODEL_DIR / "tunnelling_isolation_forest.pkl")
    state_mgr = IPStateManager()

    good, bad = 0, 0
    for raw_line in stream.read_text(encoding="utf-8").splitlines():
        try:
            event = validate_event(json.loads(raw_line))
        except Exception:
            bad += 1
            continue
        process_event(event, dga_model, tunnel_model, state_mgr)  # must not raise
        good += 1

    assert good == 2
    assert bad == 2
