"""Combine host checks, build sizes and board runs into results/summary.csv.

Usage: python -m bench.analyze
"""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from bench.config import BASELINE, PROFILES, RESULTS

KEY = ["profile", "dataset", "model", "tool"]


def board_runs() -> pd.DataFrame:
    rows = []
    for csv in sorted((RESULTS / "raw").glob("*/*.csv")):
        dataset, model, tool = csv.stem.split("__")
        info = json.loads(csv.with_suffix(".json").read_text())
        df = pd.read_csv(csv)
        mhz = int(info["mhz"])
        cycles = df["cycles_median"].to_numpy()
        rows.append({
            "profile": csv.parent.name, "dataset": dataset, "model": model, "tool": tool,
            "n_samples": len(df),
            "agreement_board": float(np.mean(df["label"] == df["ref"])),
            "accuracy_board": accuracy_score(df["y_true"], df["label"]),
            "macro_f1_board": f1_score(df["y_true"], df["label"], average="macro"),
            "cycles_median": float(np.median(cycles)),
            "cycles_p95": float(np.percentile(cycles, 95)),
            "us_median": float(np.median(cycles)) / mhz,
            "us_p95": float(np.percentile(cycles, 95)) / mhz,
            "stack_used_bytes": info["stack_size"] - info["stack_free_min"],
            "heap_free_min": info["heap_free_min"],
        })
    return pd.DataFrame(rows, columns=None if rows else KEY)


def sizes() -> pd.DataFrame:
    path = RESULTS / "size.csv"
    if not path.exists():
        return pd.DataFrame(columns=KEY)
    size = pd.read_csv(path)
    size["ram_static"] = size["dram_data"].fillna(0) + size["dram_bss"].fillna(0)
    base = (size[size.tool == BASELINE][["profile", "dataset", "bin_bytes", "ram_static"]]
            .rename(columns={"bin_bytes": "base_bin", "ram_static": "base_ram"}))
    size = size[size.tool != BASELINE].merge(base, on=["profile", "dataset"], how="left")
    size["flash_model_bytes"] = size["bin_bytes"] - size["base_bin"]
    size["ram_model_bytes"] = size["ram_static"] - size["base_ram"]
    size.loc[size.status != "ok", ["flash_model_bytes", "ram_model_bytes"]] = np.nan
    return size[KEY + ["status", "flash_model_bytes", "ram_model_bytes", "detail"]].rename(
        columns={"status": "build_status", "detail": "build_detail"})


def main() -> None:
    host = pd.read_csv(RESULTS / "host_check.csv")
    host = host.rename(columns={"status": "host_status", "agreement": "agreement_host",
                                "detail": "host_detail"})
    host = host[["dataset", "model", "tool", "host_status", "agreement_host", "host_detail"]]
    profiles = pd.DataFrame({"profile": list(PROFILES)})
    summary = profiles.merge(host, how="cross")
    summary = summary.merge(sizes(), on=KEY, how="left").merge(board_runs(), on=KEY, how="left")
    summary.sort_values(KEY).to_csv(RESULTS / "summary.csv", index=False)
    print(f"wrote {RESULTS / 'summary.csv'} ({len(summary)} rows)")


if __name__ == "__main__":
    main()
