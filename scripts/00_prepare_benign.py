import zipfile
import os

OUTPUT_FILE = "data/benign_domains.txt"
domains = set()

# 1. Extract and clean the Umbrella ZIP dataset
print("Processing Umbrella dataset...")
try:
    with zipfile.ZipFile("data/umbrella_top1m.csv.zip", "r") as z:
        # Grab the first file inside the zip (usually top-1m.csv)
        filename = z.namelist()[0] 
        with z.open(filename) as f:
            for line in f:
                # Decode bytes, strip whitespace, split by comma, grab the domain (index 1)
                domain = line.decode('utf-8').strip().split(',')[-1]
                domains.add(domain)
except FileNotFoundError:
    print("Warning: umbrella_top1m.csv.zip not found in data/ folder.")

# 2. Extract and clean the Tranco TXT dataset
print("Processing Tranco dataset...")
if os.path.exists("data/tranco_domains.txt"):
    with open("data/tranco_domains.txt", "r", encoding="utf-8") as f:
        for line in f:
            # Handle both "rank,domain" format and "domain" format safely
            domain = line.strip().split(',')[-1]
            domains.add(domain)
else:
    print("Warning: tranco_domains.txt not found in data/ folder.")

# 3. Save the final clean list
print(f"Saving {len(domains):,} unique benign domains to {OUTPUT_FILE}...")
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for d in domains:
        f.write(d + "\n")
        
print("Success! Benign dataset is ready.")