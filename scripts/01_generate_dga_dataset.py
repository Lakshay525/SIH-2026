"""
Loads your real DGArchive CSVs (extracted from the .rar) into the parquet
format config.DGA_DATASET_PATH expects: columns [domain, label, family].

Assumes each file is named like <family>_dga.csv (DGArchive's usual naming)
and has at least a 'domain' column. Adjust FILENAME_PATTERN / COLUMN_NAME
below if yours differ.
"""
import sys
import glob
import os
import re
import random
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DGA_DATASET_PATH

RAW_DIR = r"E:\Code\SIH2026\dataset\DGA"      # <-- where you extracted the .rar
FILENAME_PATTERN = "*_dga.csv"      # <-- change if your files are named differently
DOMAIN_COLUMN = "domain"            # <-- change if the CSV's domain column is named differently

MAX_PER_FAMILY = 5000               # cap huge families (chinad/conficker can be millions of rows)
                                     # so no single family dominates training / blows up runtime.
                                     # Raise this if you have time/RAM to spare.
random.seed(42)


def load_family_domains(fp: str) -> list[str]:
    """Chunked read so a multi-GB single-family CSV doesn't blow up memory."""
    domains = set()
    for chunk in pd.read_csv(fp, usecols=[DOMAIN_COLUMN], dtype=str,
                              chunksize=200_000, on_bad_lines="skip"):
        vals = chunk[DOMAIN_COLUMN].str.strip('"').str.lower().dropna().unique()
        domains.update(vals)
        if len(domains) >= MAX_PER_FAMILY * 3:   # early stop, we'll sample anyway
            break
    return list(domains)


def gen_placeholder_benign(n: int) -> list[str]:
    """Fallback benign set if you don't have real benign domains yet.
    Swap this out for real benign domains (your own DNS logs) when you have them --
    training against real benign traffic matters a lot for false-positive rate."""
    words = ["cloud","tech","data","secure","bank","market","shop","news","media","health",
             "school","travel","food","music","game","sport","finance","study","learn","connect"]
    tlds = [".com", ".net", ".org", ".in", ".co", ".io", ".edu", ".gov"]
    out = set()
    while len(out) < n:
        parts = random.sample(words, random.choice([1, 1, 2]))
        name = "".join(parts)
        if random.random() < 0.3:
            name += str(random.randint(1, 99))
        out.add(name + random.choice(tlds))
    return list(out)


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

    # -- benign side --
    # If you have a real benign domain list (recommended), load it here instead, e.g.:
    #   with open("data/benign_domains.txt") as f:
    #       benign = [l.strip().lower() for l in f if l.strip()]
    benign = gen_placeholder_benign(total_dga)
    print(f"Benign domains: {len(benign):,} (PLACEHOLDER -- swap for real benign traffic when you can)")

    domains.extend(benign)
    labels.extend([0] * len(benign))
    families.extend(["benign"] * len(benign))

    df = pd.DataFrame({"domain": domains, "label": labels, "family": families})
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    DGA_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DGA_DATASET_PATH)
    print(f"\nSaved {len(df):,} rows to {DGA_DATASET_PATH}")
    print(df["family"].value_counts())


if __name__ == "__main__":
    main()