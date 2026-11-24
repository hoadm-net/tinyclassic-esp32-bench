"""Train every model on every dataset and store the Python reference labels.

Usage: python -m bench.train [--dataset ...] [--model ...]

Output in work/models/<dataset>/:
  <model>.joblib     fitted estimator
  <model>.ref.npy    labels predicted by Python on X_test (float32 input)
Summary in results/python_models.csv.
"""

import argparse
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from bench.config import DATA_PROCESSED, DATASETS, MODELS, MODELS_DIR, RANDOM_STATE, RESULTS


def make_model(name: str, X_train: np.ndarray):
    if name == "dt5":
        return DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)
    if name == "dt10":
        return DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE)
    if name == "rf10":
        return RandomForestClassifier(n_estimators=10, max_depth=8, random_state=RANDOM_STATE)
    if name == "rf50":
        return RandomForestClassifier(n_estimators=50, max_depth=8, random_state=RANDOM_STATE)
    if name == "gnb":
        return GaussianNB()
    if name == "svc":
        # Same value as gamma="scale", passed as a number because micromlgen needs one.
        gamma = 1.0 / (X_train.shape[1] * X_train.var())
        return SVC(kernel="rbf", gamma=float(gamma), random_state=RANDOM_STATE)
    if name == "xgb":
        # base_score and num_parallel_tree must be explicit for m2cgen.
        return XGBClassifier(n_estimators=50, max_depth=4, base_score=0.5, num_parallel_tree=1,
                             random_state=RANDOM_STATE)
    raise ValueError(name)


def load_split(dataset: str):
    d = DATA_PROCESSED / dataset
    return (np.load(d / "X_train.npy"), np.load(d / "y_train.npy"),
            np.load(d / "X_test.npy"), np.load(d / "y_test.npy"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="*", default=DATASETS)
    parser.add_argument("--model", nargs="*", default=MODELS)
    args = parser.parse_args()

    out_csv = RESULTS / "python_models.csv"
    rows = pd.read_csv(out_csv).to_dict("records") if out_csv.exists() else []
    for dataset in args.dataset:
        X_train, y_train, X_test, y_test = load_split(dataset)
        out = MODELS_DIR / dataset
        out.mkdir(parents=True, exist_ok=True)
        for name in args.model:
            model = make_model(name, X_train)
            t0 = time.perf_counter()
            model.fit(X_train, y_train)
            fit_s = time.perf_counter() - t0
            ref = model.predict(X_test).astype(np.int32)
            joblib.dump(model, out / f"{name}.joblib")
            np.save(out / f"{name}.ref.npy", ref)
            row = {
                "dataset": dataset, "model": name, "fit_seconds": round(fit_s, 3),
                "accuracy": accuracy_score(y_test, ref),
                "macro_f1": f1_score(y_test, ref, average="macro"),
                "n_support_vectors": int(model.support_vectors_.shape[0]) if name == "svc" else None,
            }
            rows = [r for r in rows if not (r["dataset"] == dataset and r["model"] == name)] + [row]
            print(f"{dataset:10} {name:5} acc={row['accuracy']:.4f} f1={row['macro_f1']:.4f} fit={fit_s:.1f}s")

    RESULTS.mkdir(exist_ok=True)
    pd.DataFrame(rows).sort_values(["dataset", "model"]).to_csv(out_csv, index=False)


if __name__ == "__main__":
    main()
