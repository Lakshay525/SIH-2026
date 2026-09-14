"""
Enforces the PS's core architectural constraint -- this detector must never
resolve, connect to, or probe anything; it only classifies DNS queries it
passively observed. Adapted from the port_scanning sibling branch's
test_one_way_constraints.py: monkeypatch the socket module's syscall boundary
to raise, then run the real detection pipeline over synthetic traffic and
assert it never touches any of them.

Without a test like this, "read-only ingest" and "no active probing" are just
true by accident of what the code happens to do today -- nothing stops a
future change from quietly adding a live lookup.
"""
import socket
import sys
from pathlib import Path

import joblib
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import MODEL_DIR
from src.features.lexical import lexical_features, extract_label
from src.pipeline.state_manager import IPState, IPStateManager
from src.pipeline.engine import process_event


def _boom(*a, **k):
    raise AssertionError("pipeline code attempted a network/DNS syscall")


@pytest.fixture
def no_network(monkeypatch):
    monkeypatch.setattr(socket, "socket", _boom)
    monkeypatch.setattr(socket, "create_connection", _boom)
    monkeypatch.setattr(socket, "getaddrinfo", _boom)          # covers DNS resolution
    monkeypatch.setattr(socket, "gethostbyname", _boom, raising=False)
    monkeypatch.setattr(socket, "gethostbyaddr", _boom, raising=False)


def synthetic_events(n=200):
    events = []
    for i in range(n):
        events.append({
            "ts": i * 0.5,
            "src_ip": f"10.0.0.{i % 20}",
            "domain": f"chunk{i}abcdef0123456789.tunnel.example.com" if i % 7 == 0
                      else f"site{i}.example.org",
            "qtype": "TXT" if i % 7 == 0 else "A",
            "length": 40,
            "nxdomain": i % 11 == 0,
        })
    return events


def test_lexical_features_never_touch_network(no_network):
    """Feature extraction is pure string/math -- must never resolve anything."""
    for domain in ["google.com", "xqzplvmno-8f3ab21.net",
                    "65873f45162247cb.threatcast.guardsquare.com"]:
        lexical_features(domain)
        extract_label(domain)


def test_state_manager_never_touches_network(no_network):
    mgr = IPStateManager(max_ips=5, ttl_seconds=60)
    for event in synthetic_events():
        mgr.add_event(event["src_ip"], event["ts"], event["domain"],
                      event["qtype"], event["length"], event["nxdomain"])
    mgr.expire(now_ts=1000.0)
    IPState().stats()


def test_full_pipeline_never_touches_network(no_network):
    """The real end-to-end path: load models (disk I/O only) + classify traffic."""
    dga_model = joblib.load(MODEL_DIR / "dga_lightgbm.pkl")
    tunnel_model = joblib.load(MODEL_DIR / "tunnelling_isolation_forest.pkl")
    state_mgr = IPStateManager(max_ips=10, ttl_seconds=60)

    for event in synthetic_events():
        process_event(event, dga_model, tunnel_model, state_mgr)  # must not raise


def test_reputation_allowlist_never_touches_network(no_network):
    """
    The allowlist must resolve purely from the vendored Umbrella zip on disk.
    A reputation lookup is the most natural place for someone to later add a
    live API call or a PSL fetch -- this fails the moment that happens.
    """
    from config import DATA_DIR
    from src.features.reputation import PopularityAllowlist

    allowlist = PopularityAllowlist.from_umbrella_zip(
        DATA_DIR / "umbrella_top1m.csv.zip", top_n=5000)
    assert len(allowlist) > 0, "expected the vendored Umbrella list to load"
    for domain in ["google.com", "bf65a853.duckdns.org", "lgveufiwmnxucyym.eu"]:
        _ = domain in allowlist  # must not raise


def test_pipeline_modules_do_not_import_networking_libraries():
    """
    Static complement to the runtime checks above: the modules that make up
    the detection path should have no reason to import anything network-
    capable at all. A future PR adding `import socket`/`requests`/`dns.resolver`
    to one of these files should fail this test, not slip through review.
    """
    banned_substrings = ("import socket", "import requests", "dns.resolver",
                          "urllib.request", "http.client", "import tldextract")
    pipeline_files = [
        Path(__file__).resolve().parent.parent / "src" / "features" / "lexical.py",
        # reputation.py is the one that could most plausibly regress into a
        # live lookup -- a "reputation" layer is exactly the thing someone
        # would be tempted to wire to a remote API or to `tldextract`, which
        # fetches the Public Suffix List over the network on first use.
        Path(__file__).resolve().parent.parent / "src" / "features" / "reputation.py",
        # Same reasoning as reputation.py, more acutely: this is the module
        # that literally reimplements what tldextract does, so it's the most
        # natural place for a future edit to "simplify" by importing it.
        Path(__file__).resolve().parent.parent / "src" / "features" / "public_suffix.py",
        Path(__file__).resolve().parent.parent / "src" / "pipeline" / "state_manager.py",
        Path(__file__).resolve().parent.parent / "src" / "pipeline" / "engine.py",
        Path(__file__).resolve().parent.parent / "src" / "pipeline" / "alert_schema.py",
    ]
    for path in pipeline_files:
        text = path.read_text(encoding="utf-8")
        for banned in banned_substrings:
            assert banned not in text, f"{path} contains banned import: {banned}"
