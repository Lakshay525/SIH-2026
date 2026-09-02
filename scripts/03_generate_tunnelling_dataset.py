"""
SYNTHETIC DNS traffic windows (uses numpy.random, seeded) -- normal vs
tunnelling. Columns match src/models/tunnelling_detector.FEATURE_COLS.

TO SWAP IN REAL DATA: parse your real pcaps/logs into rows with columns
[query_rate, unique_subdomains, avg_query_len, txt_ratio, nxdomain_rate, label]
("label" = "normal" or "tunnel") and save as parquet at
config.TUNNEL_DATASET_PATH. This script then skips generation automatically.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import TUNNEL_DATASET_PATH

def main():
    if TUNNEL_DATASET_PATH.exists():
        print(f"{TUNNEL_DATASET_PATH} already exists -- skipping synthetic generation.")
        return

    rng = np.random.default_rng(42)

    n_normal = 2000
    normal = pd.DataFrame({
        "query_rate": rng.normal(8, 3, n_normal).clip(0.5, None),
        "unique_subdomains": rng.poisson(2, n_normal),
        "avg_query_len": rng.normal(18, 4, n_normal).clip(5, None),
        "txt_ratio": rng.beta(1, 30, n_normal),
        "nxdomain_rate": rng.beta(1, 20, n_normal),
        "label": "normal",
    })

    n_tunnel = 200
    tunnel = pd.DataFrame({
        "query_rate": rng.normal(40, 10, n_tunnel).clip(5, None),
        "unique_subdomains": rng.poisson(35, n_tunnel),
        "avg_query_len": rng.normal(55, 8, n_tunnel).clip(20, None),
        "txt_ratio": rng.beta(8, 3, n_tunnel),
        "nxdomain_rate": rng.beta(2, 8, n_tunnel),
        "label": "tunnel",
    })

    df = pd.concat([normal, tunnel], ignore_index=True)
    TUNNEL_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(TUNNEL_DATASET_PATH)
    print(f"Wrote {len(normal)} normal + {len(tunnel)} tunnel windows to {TUNNEL_DATASET_PATH}")

if __name__ == "__main__":
    main()
