"""
Which label the DGA classifier actually looks at.

Three separate real bugs are pinned here:
  1. Originally the FIRST label was used, so a benign CDN/tracking hostname
     ("65873f45162247cb.cdn.example.com") was judged on a random hex string
     and fired CRITICAL alerts on ordinary traffic.
  2. Fixing that by taking parts[-2] broke multi-part TLDs: "infosys.co.in"
     was judged on the constant "co". .co.in/.ac.in are in this project's
     actual deployment context.
  3. It ALSO silently broke every dynamic-DNS-based DGA family -- 9.25% of
     the DGA domains in this repo's dataset are names like
     "bf65a853.duckdns.org", where parts[-2] is the provider ("duckdns"),
     not the attacker-generated label. That fed the model a constant string
     for ~41k malicious samples and would have let a popularity allowlist
     whitelist the entire provider.
"""
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.features.lexical import extract_label, extract_registrable_domain


@pytest.mark.parametrize("domain,expected", [
    # plain
    ("google.com", "google"),
    ("aaabdxnnndiynjni.eu", "aaabdxnnndiynjni"),
    ("xqzplvmno-8f3ab21.net", "xqzplvmno-8f3ab21"),
    # (1) CDN / tracking subdomains must not be judged
    ("65873f45162247cb.threatcast.guardsquare.com", "guardsquare"),
    ("a1b2c3d4.assets.flipkart.com", "flipkart"),
    # (2) multi-part TLDs, incl. the Indian deployment context
    ("infosys.co.in", "infosys"),
    ("iitb.ac.in", "iitb"),
    ("cdn.example.co.in", "example"),
    ("www.bbc.co.uk", "bbc"),
    ("news.com.au", "news"),
    # long-tail ccTLD handled by the generic-SLD heuristic
    ("0090.co.ls", "0090"),
    ("0cug.com.cy", "0cug"),
    ("uni.ac.at", "uni"),
    # (3) dynamic DNS / user-content platforms: the ATTACKER's label
    ("bf65a853.duckdns.org", "bf65a853"),
    ("njlnehu.dyndns.org", "njlnehu"),
    ("cncdijfbm.pages.dev", "cncdijfbm"),
    ("someuser.github.io", "someuser"),
    # safety: .com is not a ccTLD, so the generic-SLD heuristic must NOT fire
    ("go.com", "go"),
    ("mysite.go.com", "go"),
    ("co.com", "co"),
    # degenerate
    ("localhost", "localhost"),
])
def test_extract_label(domain, expected):
    assert extract_label(domain) == expected


@pytest.mark.parametrize("domain,expected", [
    ("google.com", "google.com"),
    ("a1b2c3d4.assets.flipkart.com", "flipkart.com"),
    ("www.bbc.co.uk", "bbc.co.uk"),
    ("cdn.example.co.in", "example.co.in"),
    # the attacker's subdomain is part of the registrable identity here, so a
    # popularity allowlist can't whitelist the whole provider
    ("bf65a853.duckdns.org", "bf65a853.duckdns.org"),
    ("someuser.github.io", "someuser.github.io"),
])
def test_extract_registrable_domain(domain, expected):
    assert extract_registrable_domain(domain) == expected


def test_dynamic_dns_provider_is_not_a_whitelistable_identity():
    """
    The bypass this prevents: if "bf65a853.duckdns.org" collapsed to
    "duckdns.org", an attacker could host C2 on any popular dynamic-DNS
    provider and be allowlisted by construction.
    """
    assert extract_registrable_domain("bf65a853.duckdns.org") != "duckdns.org"
    assert extract_registrable_domain("evil.hopto.org") != "hopto.org"
