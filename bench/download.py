"""Download the raw datasets into data/raw and verify checksums.

Usage: python -m bench.download [--update-checksums]
"""

import argparse
import hashlib
import json
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from bench.config import CHECKSUMS, DATA_RAW

SMOKE_COMMIT = "e1b9d33cbca5885e2bb68fe4467e6e6960eebf91"

SOURCES = {
    "occupancy": "https://archive.ics.uci.edu/static/public/357/occupancy+detection.zip",
    "gasdrift": "https://archive.ics.uci.edu/static/public/224/gas+sensor+array+drift+dataset.zip",
    "har": "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip",
    "smoke": f"https://raw.githubusercontent.com/Blatts01/ai-smokedetector/{SMOKE_COMMIT}/data/final.csv",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def extract(archive: Path, dest: Path) -> None:
    with zipfile.ZipFile(archive) as z:
        z.extractall(dest)
    # HAR ships its data as a zip inside the zip.
    for inner in dest.glob("*.zip"):
        with zipfile.ZipFile(inner) as z:
            z.extractall(dest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--update-checksums", action="store_true",
                        help="record checksums of the downloaded files instead of verifying them")
    args = parser.parse_args()

    known = json.loads(CHECKSUMS.read_text()) if CHECKSUMS.exists() else {}
    for name, url in SOURCES.items():
        dest = DATA_RAW / name
        dest.mkdir(parents=True, exist_ok=True)
        filename = dest / Path(urllib.parse.unquote(url)).name.replace(" ", "_")
        if not filename.exists():
            print(f"download {name}")
            urllib.request.urlretrieve(url, filename)
        digest = sha256(filename)
        if args.update_checksums or name not in known:
            known[name] = {"file": filename.name, "url": url, "sha256": digest}
        elif known[name]["sha256"] != digest:
            raise SystemExit(f"checksum mismatch for {name}: {digest}")
        if filename.suffix == ".zip":
            extract(filename, dest)
        print(f"{name}: {digest}")

    CHECKSUMS.write_text(json.dumps(known, indent=2) + "\n")


if __name__ == "__main__":
    main()
