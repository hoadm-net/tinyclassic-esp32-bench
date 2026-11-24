"""Reproduce the host label agreement of converted models with numpy emulations.

Each emulation follows the generated C code of one tool. If it gives the same agreement
as results/host_check.csv, the stated cause explains every mismatching sample.

  rf        emlearn and micromlgen: one vote per tree (sklearn averages probabilities).
            emlearn: thresholds printed with 6 decimals ("{:.6f}f", emlearn/cgen.py), x < thr.
            micromlgen: double thresholds, x <= thr.
  dt        emlearn: thresholds printed with 6 decimals, x < thr.
  gnb       micromlgen: score_c = prior_c - sum(log var + |x - mu| / var).
  svc       micromlgen, 2 classes: sign of (sum_i a_i K(x, sv_i) - b) instead of (... + b).

Usage: python pilot/host_mismatch_emulation.py
"""

import joblib
import numpy as np
import pandas as pd

from bench.config import DATA_PROCESSED, DATASETS, MODELS_DIR, RESULTS

HOST = pd.read_csv(RESULTS / "host_check.csv").set_index(["dataset", "model", "tool"]).agreement


def six_decimals(v):
    return np.array([np.float32(float(f"{a:.6f}")) for a in v])


def tree_leaves(tree, X, thresholds, less_equal):
    node = np.zeros(len(X), dtype=int)
    rows = np.arange(len(X))
    for _ in range(tree.max_depth + 1):
        inner = tree.children_left[node] != -1
        x = X[rows, tree.feature[node].clip(0)]
        go_left = x <= thresholds[node] if less_equal else x < thresholds[node]
        child = np.where(go_left, tree.children_left[node], tree.children_right[node])
        node = np.where(inner, child, node)
    return tree.value[node][:, 0, :].argmax(1)


def emlearn_trees(model, X):
    trees = [model] if hasattr(model, "tree_") else model.estimators_
    votes = np.zeros((len(X), len(model.classes_)))
    for t in trees:
        t = t.tree_
        votes[np.arange(len(X)), tree_leaves(t, X, six_decimals(t.threshold), False)] += 1
    return votes.argmax(1)


def micromlgen_rf(model, X):
    X = X.astype(np.float64)
    votes = np.zeros((len(X), len(model.classes_)))
    for t in model.estimators_:
        t = t.tree_
        votes[np.arange(len(X)), tree_leaves(t, X, t.threshold, True)] += 1
    return votes.argmax(1)


def micromlgen_gnb(model, X):
    mu, var = model.theta_.astype(np.float32), model.var_.astype(np.float32)
    prior = model.class_prior_.astype(np.float32)
    score = [prior[c] - (np.log(var[c]).sum() + (np.abs(X - mu[c]) / var[c]).sum(1))
             for c in range(len(prior))]
    return np.stack(score, 1).argmax(1)


def micromlgen_svc_binary(model, X):
    d = model.decision_function(X) - 2 * model.intercept_[0]
    return (d > 0).astype(int)


CASES = [
    ("dt5", "emlearn", emlearn_trees), ("dt10", "emlearn", emlearn_trees),
    ("rf10", "emlearn", emlearn_trees), ("rf50", "emlearn", emlearn_trees),
    ("rf10", "micromlgen", micromlgen_rf), ("rf50", "micromlgen", micromlgen_rf),
    ("gnb", "micromlgen", micromlgen_gnb),
]


def main():
    print(f"{'dataset':10} {'model':5} {'tool':11} {'emulated':>9} {'host':>9}")
    for ds in DATASETS:
        X = np.load(DATA_PROCESSED / ds / "X_test.npy").astype(np.float32)
        cases = list(CASES)
        if len(np.unique(np.load(DATA_PROCESSED / ds / "y_test.npy"))) == 2:
            cases.append(("svc", "micromlgen", micromlgen_svc_binary))
        for model_name, tool, emulate in cases:
            model = joblib.load(MODELS_DIR / ds / f"{model_name}.joblib")
            ref = np.load(MODELS_DIR / ds / f"{model_name}.ref.npy")
            emu = np.mean(emulate(model, X) == ref)
            print(f"{ds:10} {model_name:5} {tool:11} {100 * emu:8.4f}% {100 * HOST[(ds, model_name, tool)]:8.4f}%")


if __name__ == "__main__":
    main()
