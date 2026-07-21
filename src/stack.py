"""Stack all saved OOF probability sets with a per-fold logistic meta-model,
then tune the balanced-accuracy decision rule. Uses artifacts/oof_*.npy + test_*.npy."""
import os, sys, glob, warnings
os.environ.setdefault("OMP_NUM_THREADS", "7")
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(ROOT, "artifacts")
CLASSES = ["at-risk", "unhealthy", "fit"]
C2I = {c: i for i, c in enumerate(CLASSES)}; I2C = {i: c for c, i in C2I.items()}
SEED = 42


def tune(prob, y):
    best, w = balanced_accuracy_score(y, prob.argmax(1)), np.ones(3)
    for _ in range(6):
        imp = False
        for k in range(3):
            b = w.copy(); bwv, bs = w[k], best
            for m in np.linspace(0.4, 2.5, 30):
                b[k] = m; s = balanced_accuracy_score(y, (prob * b).argmax(1))
                if s > bs: bs, bwv = s, m
            if bwv != w[k]: w[k], best, imp = bwv, bs, True
        if not imp: break
    return w, best


def main(names):
    tr = pd.read_csv(os.path.join(ROOT, "data", "train.csv"))
    te = pd.read_csv(os.path.join(ROOT, "data", "test.csv"))
    y = tr["health_condition"].map(C2I).values.astype(int)
    oofs, tests, used = [], [], []
    for n in names:
        po, pt = os.path.join(ART, f"oof_{n}.npy"), os.path.join(ART, f"test_{n}.npy")
        if os.path.exists(po) and os.path.exists(pt):
            o = np.load(po)
            if len(o) == len(y):
                oofs.append(o); tests.append(np.load(pt)); used.append(n)
    print("stacking:", used)
    for n, o in zip(used, oofs):
        print(f"  {n:14s} tuned={tune(o, y)[1]:.5f}")
    Xo = np.hstack(oofs); Xt = np.hstack(tests)
    # per-fold meta-model -> honest OOF of the stack
    meta_oof = np.zeros((len(y), 3)); meta_test = np.zeros((Xt.shape[0], 3))
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED + 1)
    for tri, vai in skf.split(Xo, y):
        m = LogisticRegression(max_iter=1000, C=10.0, class_weight="balanced")
        m.fit(Xo[tri], y[tri])
        meta_oof[vai] = m.predict_proba(Xo[vai])
        meta_test += m.predict_proba(Xt) / 5
    w, tuned = tune(meta_oof, y)
    print(f"\nSTACK tuned balanced acc = {tuned:.5f}  (multipliers {np.round(w,3)})")
    sub = pd.DataFrame({"id": te["id"], "health_condition": [I2C[i] for i in (meta_test * w).argmax(1)]})
    out = os.path.join(ROOT, "submissions", "sub_stack.csv")
    sub.to_csv(out, index=False)
    print("wrote", out, dict(sub["health_condition"].value_counts(normalize=True).round(4)))


if __name__ == "__main__":
    names = sys.argv[1:] or ["lgbm", "xgb", "histgb", "B_rule", "D_rule_orig"]
    main(names)
