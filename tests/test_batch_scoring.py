"""
The real production throughput path for DGA scoring: check_dga_batch()/
predict_batch() score many events in one booster call instead of one per
event (measured ~100x amortised vs. the single-event path -- see
predict_batch()'s docstring). Pinned here: batch output must be bit-identical
to the single-event path, not just "close enough", and allowlist filtering
must still work per-event inside a batch.
"""
import sys
from pathlib import Path

import joblib
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import MODEL_DIR
from src.pipeline.engine import check_dga, check_dga_batch


@pytest.fixture(scope="module")
def dga_model():
    return joblib.load(MODEL_DIR / "dga_lightgbm.pkl")


@pytest.fixture
def events():
    domains = ["google.com", "xqzplvmno-8f3ab21.net", "lgveufiwmnxucyym.eu",
               "myshop24.in", "bf65a853.duckdns.org", "www.bbc.co.uk"]
    return [{"ts": float(i), "src_ip": f"10.0.0.{i}", "domain": d}
            for i, d in enumerate(domains)]


def test_batch_matches_single_event_path_exactly(dga_model, events):
    single = [check_dga(dga_model, e) for e in events]
    batch = check_dga_batch(dga_model, events)

    assert len(single) == len(batch)
    for s, b in zip(single, batch):
        assert (s is None) == (b is None)
        if s is not None:
            assert s["confidence"] == pytest.approx(b["confidence"], abs=1e-9)
            assert s["flow_id"] == b["flow_id"]
            assert s["severity"] == b["severity"]


def test_batch_respects_allowlist(dga_model, events):
    class AllowAll:
        def __contains__(self, domain):
            return True

    result = check_dga_batch(dga_model, events, allowlist=AllowAll())
    assert all(a is None for a in result), "allowlisted batch should suppress every alert"


def test_batch_handles_empty_input(dga_model):
    assert check_dga_batch(dga_model, []) == []


def test_batch_output_order_matches_input_order(dga_model, events):
    batch = check_dga_batch(dga_model, events)
    assert len(batch) == len(events)
    # xqzplvmno-8f3ab21.net (index 1) and the two gibberish .eu domains
    # should fire; google.com/bbc.co.uk should not.
    assert batch[0] is None            # google.com
    assert batch[1] is not None        # xqzplvmno-8f3ab21.net
    assert batch[1]["flow_id"] == "xqzplvmno-8f3ab21.net"
