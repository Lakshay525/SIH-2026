"""
SYNTHETIC placeholder DGA dataset (uses `random`, seeded) so the whole
pipeline runs before your real dataset is ready.

TO SWAP IN YOUR REAL DATA: just save your real domains as a parquet file with
columns [domain, label, family] at config.DGA_DATASET_PATH (or change
DATA_DIR in config.py to the folder that already has it). This script will
then just skip generation and step 02 uses your file untouched.
"""
import random
import string
import hashlib
import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DGA_DATASET_PATH

random.seed(42)

WORDS = [
    "cloud", "tech", "data", "secure", "bank", "market", "shop", "news", "media", "health",
    "school", "travel", "food", "music", "game", "sport", "finance", "study", "learn", "connect",
    "global", "smart", "digital", "online", "service", "group", "system", "network", "solutions",
]
TLDS = [".com", ".net", ".org", ".in", ".co", ".io"]

def gen_benign_domain():
    n_words = random.choice([1, 1, 2])
    parts = random.sample(WORDS, n_words)
    name = "".join(parts)
    if random.random() < 0.3:
        name += str(random.randint(1, 99))
    return name + random.choice(TLDS)

def dga_conficker_style(seed):
    rnd = random.Random(seed)
    length = rnd.randint(7, 16)
    return "".join(rnd.choice(string.ascii_lowercase) for _ in range(length)) + rnd.choice(TLDS)

def dga_kraken_style(seed):
    rnd = random.Random(seed)
    vowels, consonants = "aeiou", "bcdfghjklmnpqrstvwxyz"
    length = rnd.randint(8, 14)
    s = "".join(rnd.choice(consonants) if i % 2 == 0 else rnd.choice(vowels) for i in range(length))
    return s + rnd.choice(TLDS)

def dga_hash_style(seed):
    h = hashlib.md5(str(seed).encode()).hexdigest()
    rnd = random.Random(seed)
    length = 10 + rnd.randint(0, 7)
    return h[:length] + rnd.choice(TLDS)

def main():
    if DGA_DATASET_PATH.exists():
        print(f"{DGA_DATASET_PATH} already exists -- skipping synthetic generation.")
        print("Delete it or point config.DGA_DATASET_PATH at your real file to change data.")
        return

    benign = list({gen_benign_domain() for _ in range(4000)})

    families = [dga_conficker_style, dga_kraken_style, dga_hash_style]
    domains, labels, family_names = [], [], []
    for fam in families:
        for i in range(1500):
            domains.append(fam(f"{fam.__name__}-{i}"))
            labels.append(1)
            family_names.append(fam.__name__)
    for d in benign:
        domains.append(d)
        labels.append(0)
        family_names.append("benign")

    df = pd.DataFrame({"domain": domains, "label": labels, "family": family_names})
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    DGA_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DGA_DATASET_PATH)
    print(f"Wrote {len(df)} rows ({len(benign)} benign, {len(df)-len(benign)} DGA) to {DGA_DATASET_PATH}")

if __name__ == "__main__":
    main()