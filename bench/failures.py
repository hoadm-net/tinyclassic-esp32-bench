"""List every failure and every label mismatch, with its cause.

Usage: python -m bench.failures

Reads results/host_check.csv and results/size.csv; writes results/failures.csv with one row
per (stage, profile, dataset, model, tool). Stages:
  convert    the tool did not produce code
  host       the host compiler failed or timed out
  agreement  host labels differ from Python on some test samples
  build      the ESP-IDF build failed (default flags; litpools_status gives the rebuild)
"""

import pandas as pd

from bench.config import BASELINE, RESULTS

# cause id -> (description, evidence)
CAUSES = {
    "micromlgen_xgb_binary": (
        "micromlgen ports a binary XGBClassifier with num_class=0 and fails with ZeroDivisionError",
        "meta.json error"),
    "m2cgen_svc_source_size": (
        "m2cgen writes every support vector as an expression; the HAR SVC source is about 60 MB "
        "and the host compiler does not finish in 1800 s",
        "host_check.csv detail"),
    "emlearn_threshold_6_decimals": (
        "emlearn prints split thresholds with 6 decimals ('{:.6f}f', emlearn/cgen.py) and compares x < thr",
        "pilot/host_mismatch_emulation.py"),
    "rf_hard_voting": (
        "the generated forest counts one vote per tree; scikit-learn averages class probabilities. "
        "emlearn also rounds thresholds to 6 decimals",
        "pilot/host_mismatch_emulation.py"),
    "emlearn_gnb_q16_overflow": (
        "emlearn evaluates Gaussian NB in Q16.16 fixed point, without class priors; large "
        "standardized distances and class sums overflow",
        "pilot/gnb_emlearn_q16.py (emulation within 0.6 points of host agreement)"),
    "micromlgen_gnb_formula": (
        "micromlgen scores prior - sum(log var + |x - mu| / var) instead of the Gaussian log-likelihood",
        "pilot/host_mismatch_emulation.py"),
    "micromlgen_svc_intercept_sign": (
        "micromlgen subtracts the intercept of a binary SVC instead of adding it",
        "pilot/host_mismatch_emulation.py"),
    "xgb_threshold_ties": (
        "XGBoost stores float32 thresholds and goes left on x < thr; the generated code compares with "
        "a decimal double literal (m2cgen) or with x <= thr (micromlgen), so inputs equal to a "
        "threshold can take the other branch",
        "pilot/xgboost_threshold_ties.py"),
    "xtensa_literal_range": (
        "Xtensa L32R loads a literal at most 256 KB before the instruction; the generated function is "
        "longer than that",
        "size.csv detail; litpools rebuild"),
    "region_overflow": ("the model does not fit a memory region or the app partition", "size.csv detail"),
    "build_timeout": ("the ESP-IDF build did not finish in 3600 s", "size.csv"),
    "compiler_killed": ("the compiler (cc1) was terminated by SIGKILL; cause not confirmed", "size.csv detail"),
    "unknown": ("not explained yet", ""),
}


def agreement_cause(model: str, tool: str) -> str:
    family = model.rstrip("0123456789")
    if family == "dt" and tool == "emlearn":
        return "emlearn_threshold_6_decimals"
    if family == "rf":
        return "rf_hard_voting"
    if family == "gnb":
        return {"emlearn": "emlearn_gnb_q16_overflow", "micromlgen": "micromlgen_gnb_formula"}.get(tool, "unknown")
    if family == "svc" and tool == "micromlgen":
        return "micromlgen_svc_intercept_sign"
    if family == "xgb" and tool in ("m2cgen", "micromlgen"):
        return "xgb_threshold_ties"
    return "unknown"


def host_rows() -> list[dict]:
    rows = []
    for r in pd.read_csv(RESULTS / "host_check.csv").itertuples():
        base = {"profile": "", "dataset": r.dataset, "model": r.model, "tool": r.tool}
        if r.status == "convert_error":
            cause = "micromlgen_xgb_binary" if (r.model, r.tool) == ("xgb", "micromlgen") else "unknown"
            rows.append({"stage": "convert", **base, "status": r.status, "agreement": None,
                         "detail": r.detail, "cause": cause})
        elif r.status != "ok":
            cause = "m2cgen_svc_source_size" if "timeout" in str(r.detail) and r.model == "svc" else "unknown"
            rows.append({"stage": "host", **base, "status": r.status, "agreement": None,
                         "detail": r.detail, "cause": cause})
        elif r.agreement < 1:
            n_diff = round((1 - r.agreement) * r.n_test)
            rows.append({"stage": "agreement", **base, "status": "mismatch", "agreement": r.agreement,
                         "detail": f"{n_diff} of {int(r.n_test)} samples",
                         "cause": agreement_cause(r.model, r.tool)})
    return rows


def build_rows() -> list[dict]:
    path = RESULTS / "size.csv"
    if not path.exists():
        return []
    size = pd.read_csv(path)
    if "build_flags" not in size:
        size["build_flags"] = "default"
    key = ["profile", "dataset", "model", "tool"]
    litpools = size[size.build_flags == "litpools"].set_index(key).status
    rows = []
    for r in size[(size.build_flags == "default") & (size.tool != BASELINE)].itertuples():
        if r.status in ("ok", "convert_error"):  # convert errors are listed once, from host_check
            continue
        cause = {"literal_range": "xtensa_literal_range", "too_large": "region_overflow",
                 "build_timeout": "build_timeout"}.get(r.status, "unknown")
        if cause == "unknown" and "Killed" in str(r.detail):
            cause = "compiler_killed"
        rows.append({"stage": "build", "profile": r.profile, "dataset": r.dataset, "model": r.model,
                     "tool": r.tool, "status": r.status, "agreement": None, "detail": r.detail,
                     "cause": cause,
                     "litpools_status": litpools.get((r.profile, r.dataset, r.model, r.tool))})
    return rows


def main() -> None:
    df = pd.DataFrame(host_rows() + build_rows())
    df["evidence"] = df.cause.map(lambda c: CAUSES[c][1])
    order = {"convert": 0, "host": 1, "agreement": 2, "build": 3}
    df = df.sort_values(["stage", "dataset", "model", "tool", "profile"],
                        key=lambda s: s.map(order) if s.name == "stage" else s)
    df.to_csv(RESULTS / "failures.csv", index=False)
    print(f"wrote {RESULTS / 'failures.csv'} ({len(df)} rows)\n")
    counts = df.groupby(["stage", "cause"]).size()
    for (stage, cause), n in counts.items():
        print(f"{stage:10} {cause:32} {n:4}  {CAUSES[cause][0][:90]}")


if __name__ == "__main__":
    main()
