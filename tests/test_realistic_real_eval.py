"""
scripts/03c_build_realistic_real_holdout.py fixes the ONE construction
artifact scripts/04c's ablation identified (unique_subdomains == query_rate,
which no real host produces) by sampling WITH REPLACEMENT from a small
"favourites" set per synthetic session, using real domain strings throughout.
Pinned here on a small synthetic stand-in for the real CSV (not the 5000-row
real file, so this runs fast and has no external data dependency): sessions
must show realistic repetition, not all-unique domains.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relpath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build_holdout = _load("build_realistic_holdout", "scripts/03c_build_realistic_real_holdout.py")


@pytest.fixture
def synthetic_regular_domains():
    """Stand-in for the real Mendeley 'regular' class -- a flat, distinct
    domain list, same shape as the real CSV's relevant columns."""
    n = 500
    return pd.DataFrame({
        "qname": [f"site{i}.example.com" for i in range(n)],
        "qd_qname_len": np.random.default_rng(0).integers(10, 40, n),
        "qd_qtype": 1,  # all A records, matching the real "regular" class
    })


def test_realistic_sessions_show_repeat_visits_not_all_unique(synthetic_regular_domains):
    rng = np.random.default_rng(42)
    sessions = build_holdout.build_realistic_normal_sessions(synthetic_regular_domains, rng)

    assert len(sessions) == build_holdout.N_SESSIONS
    # The whole point: unique_subdomains must be well below query_rate for
    # most sessions (a real host revisits a handful of sites), NOT equal to
    # it (which was the original scripts/03b artifact).
    ratio = sessions["unique_subdomains"] / sessions["query_rate"]
    assert ratio.mean() < 0.5, f"sessions still look all-unique: mean ratio={ratio.mean():.2f}"
    assert (sessions["unique_subdomains"] <= build_holdout.FAVOURITES_RANGE[1]).all()


def test_realistic_sessions_use_only_real_domain_strings(synthetic_regular_domains):
    rng = np.random.default_rng(1)
    sessions = build_holdout.build_realistic_normal_sessions(synthetic_regular_domains, rng)
    # avg_query_len must fall within the real pool's actual length range --
    # confirms lengths come from real strings, not a synthetic distribution.
    real_lens = synthetic_regular_domains["qd_qname_len"]
    assert sessions["avg_query_len"].min() >= real_lens.min()
    assert sessions["avg_query_len"].max() <= real_lens.max()


def test_realistic_sessions_are_labeled_normal(synthetic_regular_domains):
    rng = np.random.default_rng(2)
    sessions = build_holdout.build_realistic_normal_sessions(synthetic_regular_domains, rng)
    assert (sessions["label"] == "normal").all()
    assert (sessions["tool"] == "normal_realistic").all()
