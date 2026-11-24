"""Build train/test splits, standardise features and save them as float32.

Usage: python -m bench.split

Output per dataset in data/processed/<dataset>/:
  X_train.npy, y_train.npy, X_test.npy, y_test.npy  (X float32, y int32 in 0..K-1)
  X_test.bin  raw little-endian float32, row major (sent to the boards)
  meta.json
"""

import json

import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import LabelEncoder, StandardScaler

from bench.config import (DATA_PROCESSED, DATA_RAW, DATASETS, GASDRIFT_TRAIN_BATCHES,
                          SMOKE_SESSION_GAP_S, SMOKE_TEST_SIZE)


def load_occupancy():
    """Train on datatraining.txt, test on datatest.txt + datatest2.txt."""
    cols = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]
    raw = DATA_RAW / "occupancy"
    train = pd.read_csv(raw / "datatraining.txt")
    test = pd.concat([pd.read_csv(raw / "datatest.txt"), pd.read_csv(raw / "datatest2.txt")])
    return (train[cols].to_numpy(), train["Occupancy"].to_numpy(),
            test[cols].to_numpy(), test["Occupancy"].to_numpy(), cols)


def load_smoke():
    """Split every recording session in time: first 70 % train, last 30 % test."""
    df = pd.read_csv(DATA_RAW / "smoke" / "final.csv").sort_values("UTC").reset_index(drop=True)
    cols = [c for c in df.columns if c not in ("Fire Alarm", "CNT", "UTC")]
    session = (df["UTC"].diff().fillna(0) > SMOKE_SESSION_GAP_S).cumsum()
    position = df.groupby(session).cumcount() / session.map(session.value_counts())
    test = (position >= 1 - SMOKE_TEST_SIZE).to_numpy()
    X, y = df[cols].to_numpy(), df["Fire Alarm"].to_numpy()
    return X[~test], y[~test], X[test], y[test], cols


def load_gasdrift():
    """Train on the early batches, test on the later ones."""
    Xs, ys, batches = [], [], []
    for b in range(1, 11):
        X, y = load_svmlight_file(str(DATA_RAW / "gasdrift" / "Dataset" / f"batch{b}.dat"), n_features=128)
        Xs.append(X.toarray())
        ys.append(y.astype(int))
        batches.append(np.full(len(y), b))
    X, y, batch = np.vstack(Xs), np.concatenate(ys), np.concatenate(batches)
    train = np.isin(batch, list(GASDRIFT_TRAIN_BATCHES))
    return X[train], y[train], X[~train], y[~train], [f"f{i}" for i in range(1, 129)]


def load_har():
    """Use the subject-wise split shipped with the dataset."""
    base = DATA_RAW / "har" / "UCI HAR Dataset"
    load = lambda p: np.loadtxt(base / p)
    names = [line.split()[1] for line in (base / "features.txt").read_text().splitlines()]
    return (load("train/X_train.txt"), load("train/y_train.txt").astype(int),
            load("test/X_test.txt"), load("test/y_test.txt").astype(int), names)


LOADERS = {"occupancy": load_occupancy, "smoke": load_smoke,
           "gasdrift": load_gasdrift, "har": load_har}


def main() -> None:
    for name in DATASETS:
        X_train, y_train, X_test, y_test, features = LOADERS[name]()
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train).astype(np.float32)
        X_test = scaler.transform(X_test).astype(np.float32)
        encoder = LabelEncoder().fit(y_train)
        y_train = encoder.transform(y_train).astype(np.int32)
        y_test = encoder.transform(y_test).astype(np.int32)

        out = DATA_PROCESSED / name
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "X_train.npy", X_train)
        np.save(out / "y_train.npy", y_train)
        np.save(out / "X_test.npy", X_test)
        np.save(out / "y_test.npy", y_test)
        X_test.astype("<f4").tofile(out / "X_test.bin")
        meta = {
            "n_train": len(y_train), "n_test": len(y_test),
            "n_features": X_train.shape[1], "n_classes": len(encoder.classes_),
            "classes": encoder.classes_.tolist(), "features": features,
            "train_class_counts": np.bincount(y_train).tolist(),
            "test_class_counts": np.bincount(y_test, minlength=len(encoder.classes_)).tolist(),
        }
        (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(f"{name}: train {X_train.shape}, test {X_test.shape}, classes {meta['n_classes']}")


if __name__ == "__main__":
    main()
