# download/collect Hydro-Québec demand data into data/raw/hq_demand/

"""
Download Hydro-Québec hourly historical demand data (raw).

Output:
- data/raw/hq_demand/historique-demande-electricite-quebec.csv
- data/raw/hq_demand/historique-demande-electricite-quebec.meta.json

Run:
  python src/ingest_hq_demand.py
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen
import yaml

cfg = yaml.safe_load(Path("configs/default.yaml").read_text(encoding="utf-8"))


# --- Minimal config ---
DATASET_ID = cfg["hq"]["dataset_id"]
DOMAIN = cfg["hq"]["domain"]
TIMEZONE = cfg["project"]["timezone"]
OUT_DIR = Path("data/raw/hq_demand")
OUT_CSV = OUT_DIR / f"{DATASET_ID}.csv"
OUT_META = OUT_DIR / f"{DATASET_ID}.meta.json"

# OpenDataSoft Explore API v2.1 export endpoint (CSV)
EXPORT_URL = (
    f"https://{DOMAIN}/api/explore/v2.1/catalog/datasets/{DATASET_ID}/exports/csv"
    f"?timezone={TIMEZONE}"
)

# Basic retry
MAX_TRIES = 3
SLEEP_SECONDS = 2


def download_file(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=60) as resp:
        # Stream to disk
        with open(out_path, "wb") as f:
            while True:
                chunk = resp.read(1024 * 1024)  # 1MB
                if not chunk:
                    break
                f.write(chunk)


def main() -> None:
    print(f"Downloading: {EXPORT_URL}")
    print(f"Saving to:  {OUT_CSV}")

    last_err = None
    for attempt in range(1, MAX_TRIES + 1):
        try:
            download_file(EXPORT_URL, OUT_CSV)
            last_err = None
            break
        except (HTTPError, URLError, TimeoutError) as e:
            last_err = e
            print(f"[Attempt {attempt}/{MAX_TRIES}] Download failed: {e}")
            if attempt < MAX_TRIES:
                time.sleep(SLEEP_SECONDS)

    if last_err is not None:
        raise SystemExit(f"Failed after {MAX_TRIES} attempts: {last_err}")

    meta = {
        "dataset_id": DATASET_ID,
        "source_url": EXPORT_URL,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "file_size_bytes": OUT_CSV.stat().st_size,
    }
    OUT_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("✅ Done.")
    print(f"  - {OUT_CSV} ({meta['file_size_bytes']} bytes)")
    print(f"  - {OUT_META}")

if __name__ == "__main__":
    main()
