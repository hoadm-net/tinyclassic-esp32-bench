"""Compile every converted model for the host and compare its labels with Python.

Usage: python -m bench.host_check [--dataset ...] [--model ...] [--tool ...] [--cc cc] [--cxx c++]

Results in results/host_check.csv.
"""

import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import emlearn
import numpy as np
import pandas as pd

from bench import proc
from bench.config import (DATA_PROCESSED, HOST_DIR, MODELS_DIR, RESULTS, config_id,
                          generated_dir, iter_configs)

TIMEOUT_S = 1800


def compile_host(src_dir: Path, exe: Path, cc: str, cxx: str) -> tuple[bool, str, int]:
    """Compile adapter + model + host_main. Returns (ok, log, n_double_promotions)."""
    inc = ["-I", str(src_dir), "-I", os.path.dirname(emlearn.__file__)]
    objs, promotions = [], 0
    sources = [HOST_DIR / "host_main.c"] + sorted(src_dir.glob("*.c")) + sorted(src_dir.glob("*.cpp"))
    for i, src in enumerate(sources):
        obj = exe.parent / f"{i}.o"
        if src.suffix == ".cpp":
            compiler, flags = cxx, ["-O2", "-w", "-std=gnu++17"]
        elif src.name == "model.c":
            compiler, flags = cc, ["-O2", "-std=gnu11", "-Wdouble-promotion"]
        else:
            compiler, flags = cc, ["-O2", "-w", "-std=gnu11"]
        r = proc.run([compiler, *flags, *inc, "-c", str(src), "-o", str(obj)], timeout=TIMEOUT_S)
        if src.name == "model.c":
            promotions = r.stderr.count("[-Wdouble-promotion]")
        if r.returncode:
            return False, r.stderr[-1500:], promotions
        objs.append(str(obj))
    r = subprocess.run([cxx, *objs, "-o", str(exe), "-lm"], capture_output=True, text=True)
    if r.returncode:
        return False, r.stderr[-1500:], promotions
    return True, "", promotions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="*")
    parser.add_argument("--model", nargs="*")
    parser.add_argument("--tool", nargs="*")
    parser.add_argument("--cc", default=os.environ.get("CC", "cc"))
    parser.add_argument("--cxx", default=os.environ.get("CXX", "c++"))
    args = parser.parse_args()

    out_csv = RESULTS / "host_check.csv"
    rows = pd.read_csv(out_csv).to_dict("records") if out_csv.exists() else []
    for dataset, model, tool in iter_configs(args.dataset, args.model, args.tool):
        src_dir = generated_dir(dataset, model, tool)
        meta = json.loads((src_dir / "meta.json").read_text())
        row = {"dataset": dataset, "model": model, "tool": tool, "status": meta["status"],
               "agreement": None, "n_test": None, "double_promotions": None,
               "compile_seconds": None, "detail": meta.get("error")}
        if meta["status"] == "ok":
            with tempfile.TemporaryDirectory() as tmp:
                exe = Path(tmp) / "bench_host"
                t0 = time.perf_counter()
                try:
                    ok, log, promotions = compile_host(src_dir, exe, args.cc, args.cxx)
                except subprocess.TimeoutExpired:
                    ok, log, promotions = False, f"compile timeout after {TIMEOUT_S}s", None
                row["compile_seconds"] = round(time.perf_counter() - t0, 1)
                row["double_promotions"] = promotions if tool == "m2cgen_float" else None
                if not ok:
                    row["status"], row["detail"] = "compile_error", log.strip().splitlines()[-1][:300]
                else:
                    r = subprocess.run([str(exe), str(DATA_PROCESSED / dataset / "X_test.bin")],
                                       capture_output=True, text=True)
                    pred = np.array(r.stdout.split(), dtype=np.int32)
                    ref = np.load(MODELS_DIR / dataset / f"{model}.ref.npy")
                    if r.returncode or len(pred) != len(ref):
                        row["status"], row["detail"] = "run_error", r.stderr[-300:]
                    else:
                        row["agreement"] = float(np.mean(pred == ref))
                        row["n_test"] = len(ref)
        rows = [r for r in rows if (r["dataset"], r["model"], r["tool"]) != (dataset, model, tool)]
        rows.append(row)
        agree = "" if row["agreement"] is None else f"{100 * row['agreement']:.2f}%"
        print(f"{config_id(dataset, model, tool):32} {row['status']:14} {agree:>8}  {row['detail'] or ''}"[:200])

    RESULTS.mkdir(exist_ok=True)
    pd.DataFrame(rows).sort_values(["dataset", "model", "tool"]).to_csv(out_csv, index=False)


if __name__ == "__main__":
    main()
