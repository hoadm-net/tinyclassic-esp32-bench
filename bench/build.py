"""Build the firmware for every configuration and record flash/RAM usage.

Run inside an ESP-IDF v6.1.1 environment (after `. $IDF_PATH/export.sh`).

Usage: python -m bench.build --profile esp32_240 [--dataset ...] [--model ...] [--tool ...]

A configuration that fails on Xtensa with "l32r: literal target out of range" is built a
second time with literal pools next to the code (build_flags = "litpools").

Results in results/size.csv and results/size/<profile>/<dataset>__<model>__<tool>[__litpools].json.
"""

import argparse
import json

import pandas as pd

from bench.config import PROFILES, RESULTS, config_id, generated_dir, iter_configs
from bench.firmware import build_with_fallback

KEY = ["profile", "dataset", "model", "tool", "build_flags"]

SECTIONS = {
    "flash_text": ".flash.text",
    "flash_rodata": ".flash.rodata",
    "iram_text": ".iram0.text",
    "dram_data": ".dram0.data",
    "dram_bss": ".dram0.bss",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", nargs="+", required=True, choices=PROFILES)
    parser.add_argument("--dataset", nargs="*")
    parser.add_argument("--model", nargs="*")
    parser.add_argument("--tool", nargs="*")
    args = parser.parse_args()

    out_csv = RESULTS / "size.csv"
    rows = []
    if out_csv.exists():
        old = pd.read_csv(out_csv)
        if "build_flags" not in old:
            old["build_flags"] = "default"
        rows = old.to_dict("records")
    for profile in args.profile:
        out_dir = RESULTS / "size" / profile
        out_dir.mkdir(parents=True, exist_ok=True)
        for dataset, model, tool in iter_configs(args.dataset, args.model, args.tool,
                                                 include_baseline=True):
            meta = json.loads((generated_dir(dataset, model, tool) / "meta.json").read_text())
            if meta["status"] != "ok":
                results = [("default", {"status": meta["status"], "detail": meta.get("error")})]
            else:
                results = build_with_fallback(profile, dataset, model, tool)
            rows = [r for r in rows if (r["profile"], r["dataset"], r["model"], r["tool"])
                    != (profile, dataset, model, tool)]
            for flags, result in results:
                suffix = "" if flags == "default" else f"__{flags}"
                (out_dir / f"{dataset}__{model}__{tool}{suffix}.json").write_text(
                    json.dumps(result, indent=2) + "\n")
                row = {"profile": profile, "dataset": dataset, "model": model, "tool": tool,
                       "build_flags": flags, "status": result["status"],
                       "bin_bytes": result.get("bin_bytes"), "detail": result.get("detail")}
                sections = result.get("sections", {})
                for column, name in SECTIONS.items():
                    row[column] = sections.get(name) if sections else None
                rows.append(row)
                print(f"{profile:12} {config_id(dataset, model, tool):32} {flags:9} {row['status']:13} "
                      f"{row['bin_bytes'] or ''} {row['detail'] or ''}"[:220])
            df = pd.DataFrame(rows)
            df[KEY + [c for c in df.columns if c not in KEY]].sort_values(KEY).to_csv(out_csv, index=False)


if __name__ == "__main__":
    main()
