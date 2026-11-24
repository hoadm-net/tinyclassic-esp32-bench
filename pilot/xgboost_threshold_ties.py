"""Why do m2cgen (double) and micromlgen disagree with XGBoost on HAR?

XGBoost sends a sample left when x < threshold, with x and threshold in float32.
The generated code prints each threshold as its shortest decimal (e.g. 0.5412213):
  - m2cgen (double) compares (double)x < 0.5412213, a double that differs from the float32 value;
  - micromlgen compares x <= 0.5412213 (promoted to double) and uses <= instead of <.
Both only matter when x equals a threshold exactly, which is frequent on HAR.

This script counts such ties and recompiles patched copies of the generated code.
Run from the repository root after bench.convert:  python pilot/xgboost_threshold_ties.py
"""

import json
import re
import subprocess
import tempfile
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def tie_rate(dataset):
    model = joblib.load(ROOT / "work/models" / dataset / "xgb.joblib")
    X = np.load(ROOT / "data/processed" / dataset / "X_test.npy")
    thresholds = {}

    def walk(node):
        if "leaf" in node:
            return
        thresholds.setdefault(int(node["split"][1:]), set()).add(np.float32(node["split_condition"]))
        for child in node["children"]:
            walk(child)

    for tree in model.get_booster().get_dump(dump_format="json"):
        walk(json.loads(tree))
    tie = np.zeros(len(X), bool)
    for feature, values in thresholds.items():
        tie |= np.isin(X[:, feature], np.array(sorted(values), dtype=np.float32))
    return tie.mean()


def agreement(dataset, tool, patch):
    src = ROOT / "generated" / dataset / "xgb" / tool
    ref = np.load(ROOT / "work/models" / dataset / "xgb.ref.npy")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for f in src.iterdir():
            if f.suffix in (".c", ".cpp", ".h"):
                text = f.read_text()
                (tmp / f.name).write_text(patch(text) if f.name.startswith("model.") else text)
        objs = []
        for i, f in enumerate([ROOT / "host/host_main.c", *tmp.glob("*.c"), *tmp.glob("*.cpp")]):
            obj = tmp / f"{i}.o"
            compiler = "c++" if f.suffix == ".cpp" else "cc"
            subprocess.run([compiler, "-O2", "-w", "-I", str(tmp), "-c", str(f), "-o", str(obj)], check=True)
            objs.append(str(obj))
        subprocess.run(["c++", *objs, "-o", str(tmp / "exe")], check=True)
        out = subprocess.run([str(tmp / "exe"), str(ROOT / "data/processed" / dataset / "X_test.bin")],
                             capture_output=True, text=True, check=True).stdout
    return np.mean(np.array(out.split(), dtype=int) == ref)


NUM = r"(-?\d+\.\d+(?:[eE][-+]?\d+)?)"
PATCHES = {
    "m2cgen": {
        "as generated": lambda s: s,
        "float32 thresholds": lambda s: re.sub(r"(input\[\d+\] < )" + NUM, r"\1(double)\2f", s),
    },
    "micromlgen": {
        "as generated": lambda s: s,
        "float32 thresholds": lambda s: re.sub(r"(x\[\d+\] <= )" + NUM, r"\1\2f", s),
        "float32 thresholds and <": lambda s: re.sub(r"(x\[\d+\]) <= " + NUM, r"\1 < \2f", s),
    },
}


def main():
    for dataset in ["occupancy", "smoke", "gasdrift", "har"]:
        print(f"{dataset}: {tie_rate(dataset):.2%} of test samples tie with a split threshold")
        for tool, patches in PATCHES.items():
            if not (ROOT / "generated" / dataset / "xgb" / tool / "model_info.h").exists():
                continue
            meta = json.loads((ROOT / "generated" / dataset / "xgb" / tool / "meta.json").read_text())
            if meta["status"] != "ok":
                print(f"  {tool:11} {meta['status']}")
                continue
            for name, patch in patches.items():
                print(f"  {tool:11} {name:26} {agreement(dataset, tool, patch):.2%}")


if __name__ == "__main__":
    main()
