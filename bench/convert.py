"""Convert trained models to C/C++ with each tool.

Usage: python -m bench.convert [--dataset ...] [--model ...] [--tool ...]

Every configuration gets a directory generated/<dataset>/<model>/<tool>/ with
  model_info.h   BENCH_N_FEATURES, BENCH_N_CLASSES, BENCH_MODEL_ID
  model.h/.c     raw output of the tool
  adapter.c/.cpp defines `int bench_predict(const float *x)` (C linkage)
  meta.json      conversion status
The firmware and the host check compile the same files.
"""

import argparse
import json
import re
import subprocess
import sys
import threading
import time
import traceback

import emlearn
import joblib
import m2cgen
import micromlgen

from bench.config import (BASELINE, DATA_PROCESSED, MODELS_DIR, config_id, generated_dir,
                          iter_configs)

# m2cgen recurses deeply on large SVC models; conversions run on a thread with a large stack.
sys.setrecursionlimit(1_000_000)
THREAD_STACK_BYTES = 512 * 1024 * 1024

C_HEADERS = "#include <stdint.h>\n#include <string.h>\n#include <math.h>\n"


def model_info(dataset, model, tool, n_features, n_classes):
    return (f"#pragma once\n"
            f"#define BENCH_N_FEATURES {n_features}\n"
            f"#define BENCH_N_CLASSES {n_classes}\n"
            f"#define BENCH_MODEL_ID \"{config_id(dataset, model, tool)}\"\n")


def write_baseline(out):
    (out / "adapter.c").write_text(
        '#include "model_info.h"\n\n'
        "int bench_predict(const float *x)\n{\n    (void)x;\n    return 0;\n}\n")
    return {"lang": "c"}


def write_emlearn(est, model, out, n_features, n_classes):
    if model.startswith(("dt", "rf")):
        converted = emlearn.convert(est, method="inline", dtype="float")
    else:
        converted = emlearn.convert(est)
    converted.save(file=str(out / "model.h"), name="model")
    (out / "adapter.c").write_text(
        C_HEADERS + '#include "model_info.h"\n#include "model.h"\n\n'
        "int bench_predict(const float *x)\n{\n"
        "    return (int)model_predict(x, BENCH_N_FEATURES);\n}\n")
    return {"lang": "c"}


def write_micromlgen(est, model, out, n_features, n_classes):
    if model == "gnb" and not hasattr(est, "sigma_"):
        est.sigma_ = est.var_  # attribute renamed in scikit-learn 1.0
    (out / "model.h").write_text(micromlgen.port(est, classname="Model"))
    (out / "adapter.cpp").write_text(
        C_HEADERS + '#include <stdlib.h>\n#include "model_info.h"\n#include "model.h"\n\n'
        "static Eloquent::ML::Port::Model bench_model;\n\n"
        'extern "C" int bench_predict(const float *x)\n{\n'
        "    return bench_model.predict(const_cast<float *>(x));\n}\n")
    return {"lang": "cpp"}


def m2cgen_to_float(src: str) -> str:
    """Rewrite m2cgen C output to single precision."""
    src = re.sub(r"\bdouble\b", "float", src)
    src = re.sub(r"\bexp\(", "expf(", src)
    src = re.sub(r"\bpow\(", "powf(", src)
    src = re.sub(r"(\d\.\d+(?:[eE][-+]?\d+)?)(?![\dfF])", r"\1f", src)
    return src


def write_m2cgen(est, model, out, n_features, n_classes, use_float=False):
    src = m2cgen.export_to_c(est)
    if use_float:
        src = m2cgen_to_float(src)
    T = "float" if use_float else "double"
    (out / "model.c").write_text(C_HEADERS + src)

    if re.search(r"\bvoid score\(", src):
        if model == "svc" and n_classes > 2:
            n_out, decode = n_classes * (n_classes - 1) // 2, "ovo"
        else:
            n_out, decode = n_classes, "argmax"
        proto = f"void score({T} *input, {T} *output);"
    else:
        n_out, decode = 1, "sign"
        proto = f"{T} score({T} *input);"

    if use_float:
        prep, arg = "", "(float *)x"
    else:
        prep = ("    double in[BENCH_N_FEATURES];\n"
                "    for (int i = 0; i < BENCH_N_FEATURES; i++) in[i] = x[i];\n")
        arg = "in"

    if decode == "sign":
        body = f"    return score({arg}) > 0 ? 1 : 0;\n"
    elif decode == "argmax":
        body = (f"    {T} out[{n_out}];\n    score({arg}, out);\n"
                "    int best = 0;\n"
                f"    for (int c = 1; c < {n_out}; c++) if (out[c] > out[best]) best = c;\n"
                "    return best;\n")
    else:  # one-vs-one voting, pairs ordered (0,1), (0,2), ..., (K-2,K-1)
        body = (f"    {T} out[{n_out}];\n    score({arg}, out);\n"
                "    int votes[BENCH_N_CLASSES] = {0};\n"
                "    int k = 0;\n"
                "    for (int i = 0; i < BENCH_N_CLASSES; i++)\n"
                "        for (int j = i + 1; j < BENCH_N_CLASSES; j++, k++)\n"
                "            votes[out[k] > 0 ? i : j]++;\n"
                "    int best = 0;\n"
                "    for (int c = 1; c < BENCH_N_CLASSES; c++) if (votes[c] > votes[best]) best = c;\n"
                "    return best;\n")

    (out / "adapter.c").write_text(
        '#include "model_info.h"\n\n' + proto + "\n\n"
        "int bench_predict(const float *x)\n{\n" + prep + body + "}\n")
    return {"lang": "c", "decode": decode, "n_outputs": n_out}


def convert_one(dataset, model, tool):
    meta_ds = json.loads((DATA_PROCESSED / dataset / "meta.json").read_text())
    n_features, n_classes = meta_ds["n_features"], meta_ds["n_classes"]
    out = generated_dir(dataset, model, tool)
    out.mkdir(parents=True, exist_ok=True)
    for f in out.iterdir():
        f.unlink()
    (out / "model_info.h").write_text(model_info(dataset, model, tool, n_features, n_classes))

    meta = {"dataset": dataset, "model": model, "tool": tool,
            "n_features": n_features, "n_classes": n_classes}
    t0 = time.perf_counter()
    try:
        if tool == BASELINE:
            meta.update(write_baseline(out))
        else:
            est = joblib.load(MODELS_DIR / dataset / f"{model}.joblib")
            if tool == "emlearn":
                meta.update(write_emlearn(est, model, out, n_features, n_classes))
            elif tool == "micromlgen":
                meta.update(write_micromlgen(est, model, out, n_features, n_classes))
            elif tool == "m2cgen":
                meta.update(write_m2cgen(est, model, out, n_features, n_classes))
            elif tool == "m2cgen_float":
                meta.update(write_m2cgen(est, model, out, n_features, n_classes, use_float=True))
            else:
                raise ValueError(tool)
        meta["status"] = "ok"
    except Exception as e:  # conversion failures are results, not crashes
        meta["status"] = "convert_error"
        meta["error"] = f"{type(e).__name__}: {e}"[:500]
        meta["traceback"] = traceback.format_exc()[-2000:]
        for f in out.iterdir():
            if f.name not in ("model_info.h",):
                f.unlink()
    meta["seconds"] = round(time.perf_counter() - t0, 3)
    meta["source_bytes"] = sum(f.stat().st_size for f in out.iterdir() if f.name.startswith("model."))
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def convert_in_thread(dataset, model, tool):
    """Run one conversion on a thread with a large stack (m2cgen recurses deeply)."""
    threading.stack_size(THREAD_STACK_BYTES)
    worker = threading.Thread(target=convert_one, args=(dataset, model, tool))
    worker.start()
    worker.join()


def convert_isolated(dataset, model, tool):
    """Convert in a child process so that a crash of the tool is recorded, not fatal."""
    meta_path = generated_dir(dataset, model, tool) / "meta.json"
    meta_path.unlink(missing_ok=True)
    r = subprocess.run([sys.executable, "-m", "bench.convert", "--one", dataset, model, tool],
                       capture_output=True, text=True)
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    meta = {"dataset": dataset, "model": model, "tool": tool, "status": "convert_error",
            "error": f"converter process exited with code {r.returncode}",
            "stderr": r.stderr[-2000:], "seconds": None, "source_bytes": 0}
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="*")
    parser.add_argument("--model", nargs="*")
    parser.add_argument("--tool", nargs="*")
    parser.add_argument("--one", nargs=3, metavar=("DATASET", "MODEL", "TOOL"), help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.one:
        convert_in_thread(*args.one)
        return

    for dataset, model, tool in iter_configs(args.dataset, args.model, args.tool,
                                             include_baseline=args.model is None and args.tool is None):
        meta = convert_isolated(dataset, model, tool)
        detail = meta.get("error", f"{meta['source_bytes'] // 1024} KB")
        seconds = "" if meta["seconds"] is None else f"{meta['seconds']:.1f}s"
        print(f"{config_id(dataset, model, tool):32} {meta['status']:14} {seconds:>8}  {detail}")


if __name__ == "__main__":
    main()
