"""
Builds a REAL-DATA HELD-OUT EVALUATION set for the tunnelling detector --
never used for training, only to honestly check how the synthetic-trained
Isolation Forest (scripts/03 + scripts/04) generalizes to real DNS-tunnelling
query traffic. Mirrors the honesty pattern already used for the DGA
classifier's held-out-family eval (src/models/dga_lightgbm.held_out_family_eval)
and the sibling data-exfiltration branch's CIC-IDS2018 holdout check.

Source: "DNS Tunneling Queries for Binary Classification" (despite the name,
this specific file is the *multilabel* variant), Yakov Bubnov, Mendeley Data,
DOI 10.17632/mzn9hvdcxg.2, CC BY 4.0. One row per observed DNS query with
label 0=regular, 1=dns2tcp, 2=dnscapy, 3=iodine, 4=tuns.

This dataset is per-query, not per-window, so we group same-label rows into
synthetic "sessions" (shuffled, chunked) to approximate what one source IP's
60-second window would look like, and compute the same five aggregate
features the model was trained on -- straight from the real query strings
and real recorded qd_qtype, not assumed.

Known, load-bearing limitation (see README "Limitations"): this dataset has
no rcode/answer-outcome field usable as a real NXDOMAIN signal (the "regular"
class has ancount==0 for every single row, which is a recording artifact, not
a real NXDOMAIN pattern -- using it would mislabel 100% of benign sessions as
all-NXDOMAIN). nxdomain_rate is therefore held at a neutral 0.0 for every
real-derived session below; this eval does not exercise that feature.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR

RAW_CSV = DATA_DIR / "real_tunnel_domains.csv"
OUT_PATH = DATA_DIR / "real_tunnel_eval_windows.parquet"
TXT_QTYPE = 16  # DNS TYPE 16 = TXT

TOOL_MAP = {0: "normal", 1: "dns2tcp", 2: "dnscapy", 3: "iodine", 4: "tuns"}
# Real per-session query counts differ a lot by label (normal traffic is
# chattier in short bursts than a single tunnel session tends to be sampled
# here) -- sized so each group yields a reasonable number of eval windows.
SESSION_SIZE = {"normal": 12, "dns2tcp": 40, "dnscapy": 40, "iodine": 40, "tuns": 40}
SEED = 42


def make_sessions(rows: pd.DataFrame, tool: str, session_size: int, rng) -> pd.DataFrame:
    idx = rows.index.to_numpy().copy()
    rng.shuffle(idx)
    windows = []
    for start in range(0, len(idx) - session_size + 1, session_size):
        chunk = rows.loc[idx[start:start + session_size]]
        windows.append({
            "query_rate": len(chunk),
            "unique_subdomains": chunk["qname"].nunique(),
            "avg_query_len": chunk["qd_qname_len"].mean(),
            "txt_ratio": (chunk["qd_qtype"] == TXT_QTYPE).mean(),
            "nxdomain_rate": 0.0,  # see module docstring -- not derivable from this dataset
            "label": "normal" if tool == "normal" else "tunnel",
            "tool": tool,
        })
    return pd.DataFrame(windows)


def main():
    if not RAW_CSV.exists():
        print(f"{RAW_CSV} not found -- run this after downloading the real dataset.")
        return

    df = pd.read_csv(RAW_CSV)
    rng = np.random.default_rng(SEED)

    all_windows = []
    for label_val, tool in TOOL_MAP.items():
        rows = df[df["label"] == label_val]
        windows = make_sessions(rows, tool, SESSION_SIZE[tool], rng)
        print(f"{tool:8s}: {len(rows):5,} real queries -> {len(windows):3d} sessions")
        all_windows.append(windows)

    out = pd.concat(all_windows, ignore_index=True)
    out.to_parquet(OUT_PATH)
    print(f"\nSaved {len(out):,} real-derived holdout windows to {OUT_PATH}")
    print(out.groupby(["label", "tool"])[["avg_query_len", "txt_ratio"]].mean())


if __name__ == "__main__":
    main()
