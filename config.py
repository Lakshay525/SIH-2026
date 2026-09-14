"""
Central config. Point DATA_DIR / the *_DATASET_PATH constants at your real
data and every script below just works, unchanged.
"""
from pathlib import Path

DATA_DIR = Path("data")          # <-- CHANGE THIS to your dataset's folder

# Expected shape if you bring your own file (parquet or csv, either works
# with pandas.read_parquet/read_csv — just change the one read call in
# scripts/02 and scripts/04 if your real data is .csv instead of .parquet):
#   DGA_DATASET_PATH    -> columns: domain (str), label (0/1), family (str)
#   TUNNEL_DATASET_PATH -> columns: query_rate, unique_subdomains,
#                          avg_query_len, txt_ratio, nxdomain_rate, label (str)
DGA_DATASET_PATH = DATA_DIR / "dga_dataset.parquet"
TUNNEL_DATASET_PATH = DATA_DIR / "tunnelling_windows.parquet"

WORDLIST_PATH = DATA_DIR / "english_words.txt"   # any 1-word-per-line dict file
MODEL_DIR = DATA_DIR / "models"

# Vendored snapshot of https://publicsuffix.org/list/public_suffix_list.dat
# (MPL-2.0). See src/features/public_suffix.py for why this is a static file
# and not a live-fetching library.
PUBLIC_SUFFIX_LIST_PATH = DATA_DIR / "public_suffix_list.dat"

# Raw inputs consumed by scripts/01_generate_dga_dataset.py
DGA_RAW_DIR = Path(r"E:\Code\SIH2026\dataset\DGA")       # <-- per-family DGArchive CSVs
BENIGN_DOMAINS_PATH = DATA_DIR / "benign_domains.txt"    # <-- merged benign domain list