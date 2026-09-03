"""
Loads real DGArchive CSVs into config.DGA_DATASET_PATH's parquet format:
columns [domain, label, family].

Benign domains: loaded directly from the Tranco top-1m.csv (rank,domain
format, no header). Falls back to a SAFE synthetic generator (won't freeze
even at hundreds of thousands of rows) if the file is missing or too small.
"""
import sys
import glob
import os
import re
import random
import string
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DGA_DATASET_PATH

RAW_DIR = r"E:\Code\SIH2026\dataset\DGA"                              # <-- DGArchive CSVs
FILENAME_PATTERN = "*_dga.csv"
DOMAIN_COLUMN = "domain"
BENIGN_FILE = r"E:\Code\SIH2026\SIH-2026\data\benign_domains.txt"    # <-- New merged file   

MAX_PER_FAMILY = 5000   # cap huge families so none dominates training
random.seed(42)


def load_family_domains(fp: str) -> list[str]:
    """Chunked read so a multi-GB single-family CSV doesn't blow up memory."""
    domains = set()
    for chunk in pd.read_csv(fp, usecols=[DOMAIN_COLUMN], dtype=str,
                              chunksize=200_000, on_bad_lines="skip"):
        vals = chunk[DOMAIN_COLUMN].str.strip('"').str.lower().dropna().unique()
        domains.update(vals)
        if len(domains) >= MAX_PER_FAMILY * 3:
            break
    return list(domains)


def gen_synthetic_benign(n: int) -> list[str]:
    """
    SAFE synthetic fallback -- per-item random seed + random alnum suffix
    guarantees n unique strings in one pass, so it can never freeze in an
    infinite retry loop no matter how large n is.
    """
    words = ["cloud", "tech", "data", "secure", "bank", "market", "shop", "news", "media",
             "health", "school", "travel", "food", "music", "game", "sport", "finance",
             "study", "learn", "connect", "global", "smart", "digital", "online", "service",
             "group", "system", "network", "solutions", "trade", "city", "home", "life",
             "world", "team", "hub", "zone", "express", "direct", "central", "prime", "core"]
    tlds = [".com", ".net", ".org", ".in", ".co", ".io", ".edu", ".gov"]

    out = []
    for i in range(n):
        rnd = random.Random(f"benign-{i}")
        parts = rnd.sample(words, rnd.choice([1, 2]))
        name = "".join(parts)
        suffix = "".join(rnd.choice(string.ascii_lowercase + string.digits) for _ in range(4))
        out.append(f"{name}{suffix}{rnd.choice(tlds)}")
    return out


def load_real_benign(n: int, path: str = BENIGN_FILE) -> list[str] | None:
    """Returns n real benign domains from a Tranco-format CSV (rank,domain),
    or None if the file doesn't exist."""
    p = Path(path)
    if not p.exists():
        print(f"[benign] {path} not found -- will use synthetic fallback.")
        print("[benign] Download a real list, e.g. Tranco: https://tranco-list.eu/")
        return None

    df = pd.read_csv(p, header=None, names=["domain"], dtype=str)
    all_domains = df["domain"].str.strip().str.lower().dropna().unique().tolist()
    print(f"[benign] Loaded {len(all_domains):,} real domains from {path}")

    if len(all_domains) < n:
        print(f"[benign] Only {len(all_domains):,} available, need {n:,} -- "
              f"topping up shortfall with synthetic domains.")
        random.shuffle(all_domains)
        shortfall = n - len(all_domains)
        return all_domains + gen_synthetic_benign(shortfall)

    random.shuffle(all_domains)
    return all_domains[:n]


def main():
    files = sorted(glob.glob(f"{RAW_DIR}/{FILENAME_PATTERN}"))
    print(f"Found {len(files)} family CSVs in {RAW_DIR}")
    if not files:
        print(f"No files matched {RAW_DIR}/{FILENAME_PATTERN} -- check the folder/pattern.")
        return

    domains, labels, families = [], [], []
    for fp in files:
        family = re.sub(r'^\d+_', '', os.path.basename(fp)).replace('_dga.csv', '')
        fam_domains = load_family_domains(fp)
        if len(fam_domains) > MAX_PER_FAMILY:
            fam_domains = random.sample(fam_domains, MAX_PER_FAMILY)
        print(f"{family:>20}: {len(fam_domains):>7,} domains kept")
        domains.extend(fam_domains)
        labels.extend([1] * len(fam_domains))
        families.extend([family] * len(fam_domains))

    total_dga = len(domains)
    print(f"\nTotal DGA domains: {total_dga:,} across {len(files)} families")

    benign = load_real_benign(total_dga)
    if benign is None:
        print(f"[benign] Generating {total_dga:,} synthetic benign domains...")
        benign = gen_synthetic_benign(total_dga)
    print(f"Benign domains: {len(benign):,}")

    domains.extend(benign)
    labels.extend([0] * len(benign))
    families.extend(["benign"] * len(benign))

    print("Building DataFrame...")
    df = pd.DataFrame({"domain": domains, "label": labels, "family": families})

    print("Shuffling DataFrame...")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print("Saving to Parquet...")
    DGA_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DGA_DATASET_PATH)

    print(f"\nSaved {len(df):,} rows to {DGA_DATASET_PATH}")
    print(df["family"].value_counts())


if __name__ == "__main__":
    main()