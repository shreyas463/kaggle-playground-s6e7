"""Fine decision-rule search on the best blend. Balanced accuracy only cares about
the per-class multipliers on the stacked probabilities; search them finely."""
import os, sys, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(ROOT, "artifacts")
C2I = {"at-risk": 0, "unhealthy": 1, "fit": 2}; I2C = {v: k for k, v in C2I.items()}


def bacc(y, pred):
    return np.mean([(pred[y == k] == k).mean() for k in range(3)])


def main(names):
    tr = pd.read_csv(os.path.join(ROOT, "data", "train.csv"))
    te = pd.read_csv(os.path.join(ROOT, "data", "test.csv"))
    y = tr["health_condition"].map(C2I).values
    oofs = [np.load(f"{ART}/oof_{n}.npy") for n in names]
    tests = [np.load(f"{ART}/test_{n}.npy") for n in names]
    Xo, Xt = np.hstack(oofs), np.hstack(tests)
    meta_oof = np.zeros((len(y), 3)); meta_test = np.zeros((len(Xt), 3))
    for tri, vai in StratifiedKFold(5, shuffle=True, random_state=44).split(Xo, y):
        m = LogisticRegression(max_iter=2000, C=10, class_weight="balanced").fit(Xo[tri], y[tri])
        meta_oof[vai] = m.predict_proba(Xo[vai]); meta_test += m.predict_proba(Xt) / 5
    print("blend:", names, flush=True)
    # fine 2D search (scale-invariant: fix w_at-risk = 1)
    grid = np.linspace(0.3, 2.2, 96)
    best, bw = bacc(y, meta_oof.argmax(1)), (1.0, 1.0, 1.0)
    for w1 in grid:
        col = meta_oof.copy(); col[:, 1] *= w1
        for w2 in grid:
            c2 = col.copy(); c2[:, 2] *= w2
            s = bacc(y, c2.argmax(1))
            if s > best: best, bw = s, (1.0, w1, w2)
    print(f"fine-tuned OOF balanced acc = {best:.5f}  multipliers={np.round(bw,3)}", flush=True)
    lab = (meta_test * np.array(bw)).argmax(1)
    sub = pd.DataFrame({"id": te["id"], "health_condition": [I2C[i] for i in lab]})
    out = os.path.join(ROOT, "submissions", "sub_fine.csv"); sub.to_csv(out, index=False)
    print("wrote", out, dict(sub["health_condition"].value_counts(normalize=True).round(4)), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["lgbm", "xgb", "histgb", "B_rule", "tabpfn"])
