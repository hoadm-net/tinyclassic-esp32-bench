"""Where does emlearn's Gaussian NB lose agreement with scikit-learn?

Re-implements eml_bayes_predict (emlearn 0.23.2, eml_bayes.h) three ways and compares
each with scikit-learn on the processed datasets:
  1. float, exact log-pdf, no class prior
  2. float, emlearn's quadratic log2-pdf approximation, no class prior
  3. int32 Q16.16 arithmetic exactly as in eml_bayes.h (wrap-around on overflow)
Run from the repository root after bench.split and bench.train:
  python pilot/gnb_emlearn_q16.py
"""

import joblib
import numpy as np

ONE = 1 << 16
A, B, C = -0.7213475204444817, -1.0005845727355313e-15, -1.3257480647361592


def q16(x):
    return np.asarray(np.asarray(x, dtype=np.float64) * ONE).astype(np.int64).astype(np.int32)


def q16_mul(x, y):  # ((x) >> 8) * ((y) >> 8), int32 wrap-around
    return ((x.astype(np.int64) >> 8) * (y.astype(np.int64) >> 8)).astype(np.int32)


def q16_div(a, b):
    temp = a.astype(np.int64) << 16
    same_sign = ((temp >= 0) & (b >= 0)) | ((temp < 0) & (b < 0))
    temp = np.where(same_sign, temp + b // 2, temp - b // 2)
    return np.trunc(temp / b).astype(np.int64).astype(np.int32)


def log2_fix(x, precision=16):
    """eml_bayes_log2fix, element-wise (only used once per summary, at conversion time)."""
    return np.log2(x / ONE)


def predict_q16(model, X):
    mean = q16(model.theta_)
    std = q16(np.sqrt(model.var_))
    stdlog2 = q16(np.log2(np.sqrt(model.var_)))
    a, b, c = q16(A), q16(B), q16(C)
    scores = np.zeros((len(X), len(model.classes_)), dtype=np.int32)
    xq = q16(X)
    for k in range(len(model.classes_)):
        xm = q16_div(xq - mean[k], np.broadcast_to(std[k], xq.shape))
        xx = np.abs(xm)
        p = q16_mul(a, q16_mul(xx, xx)) + q16_mul(b, xx) + c - stdlog2[k]
        scores[:, k] = p.astype(np.int32).sum(axis=1, dtype=np.int64).astype(np.int32)  # wraps like int32
    return scores.argmax(axis=1)


def predict_float(model, X, approx):
    std = np.sqrt(model.var_)
    scores = []
    for k in range(len(model.classes_)):
        xm = np.abs((X - model.theta_[k]) / std[k])
        if approx:
            p = A * xm ** 2 + B * xm + C - np.log2(std[k])
        else:
            p = -0.5 * xm ** 2 - np.log(std[k])
        scores.append(p.sum(axis=1))
    return np.argmax(np.stack(scores, axis=1), axis=1)


def main():
    for ds in ["occupancy", "smoke", "gasdrift", "har"]:
        X = np.load(f"data/processed/{ds}/X_test.npy").astype(np.float64)
        model = joblib.load(f"work/models/{ds}/gnb.joblib")
        ref = model.predict(X.astype(np.float32))
        exact = predict_float(model, X, approx=False)
        approx = predict_float(model, X, approx=True)
        fixed = predict_q16(model, X)
        print(f"{ds:10} no prior {np.mean(exact == ref):7.2%}   "
              f"+ quadratic approx {np.mean(approx == ref):7.2%}   "
              f"+ Q16.16 {np.mean(fixed == ref):7.2%}")


if __name__ == "__main__":
    main()
