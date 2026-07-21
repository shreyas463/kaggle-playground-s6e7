"""Fast, apples-to-apples comparison of algorithms on a stratified subsample.
Reports OOF balanced accuracy for each. (Submission uses the full-data GBMs.)"""
import os, sys, time, warnings
os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import build_matrices, C2I

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 42
N = 200_000


def bw(y):
    c = np.bincount(y, minlength=3)
    return (len(y) / (3 * c))[y]


def cv(fit_predict, y):
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    oof = np.zeros((len(y), 3))
    for tri, vai in skf.split(np.zeros(len(y)), y):
        oof[vai] = fit_predict(tri, vai)
    return balanced_accuracy_score(y, oof.argmax(1))


def main():
    tr = pd.read_csv(os.path.join(ROOT, "data", "train.csv"))
    y_full = tr["health_condition"].map(C2I).values.astype(int)
    # stratified subsample
    rng = np.random.RandomState(SEED)
    idx = np.concatenate([rng.choice(np.where(y_full == c)[0],
                          int(N * (y_full == c).mean()), replace=False) for c in range(3)])
    rng.shuffle(idx)
    sub = tr.iloc[idx].reset_index(drop=True)
    y = y_full[idx]
    M = build_matrices(sub, sub.head(5))  # test unused here
    Xn, _, cat_idx, cat_cols, num_cols = M["native"]
    Xe, _ = M["enc"]; Xs, _ = M["scaled"]
    Xe = Xe.values

    from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                                  HistGradientBoostingClassifier, GradientBoostingClassifier)
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    import lightgbm as lgb
    from xgboost import XGBClassifier

    def lgbm(tri, vai):
        d = lgb.Dataset(Xn.iloc[tri], y[tri], weight=bw(y[tri]), categorical_feature=cat_idx)
        m = lgb.train(dict(objective="multiclass", num_class=3, learning_rate=0.05,
                           num_leaves=63, verbose=-1, seed=SEED), d, num_boost_round=300)
        return m.predict(Xn.iloc[vai])

    def xgb(tri, vai):
        m = XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=7, subsample=0.8,
                          tree_method="hist", enable_categorical=True, n_jobs=6, random_state=SEED)
        m.fit(Xn.iloc[tri], y[tri], sample_weight=bw(y[tri]))
        return m.predict_proba(Xn.iloc[vai])

    def histgb(tri, vai):
        m = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=400, max_leaf_nodes=63,
                                           categorical_features="from_dtype",
                                           class_weight="balanced", random_state=SEED)
        m.fit(Xn.iloc[tri], y[tri]); return m.predict_proba(Xn.iloc[vai])

    def sk_gbm(tri, vai):
        m = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=SEED)
        m.fit(Xe[tri], y[tri], sample_weight=bw(y[tri])); return m.predict_proba(Xe[vai])

    def rf(tri, vai):
        m = RandomForestClassifier(n_estimators=300, max_depth=18, min_samples_leaf=20,
                                   class_weight="balanced", n_jobs=6, random_state=SEED)
        m.fit(Xe[tri], y[tri]); return m.predict_proba(Xe[vai])

    def et(tri, vai):
        m = ExtraTreesClassifier(n_estimators=400, max_depth=22, min_samples_leaf=15,
                                 class_weight="balanced", n_jobs=6, random_state=SEED)
        m.fit(Xe[tri], y[tri]); return m.predict_proba(Xe[vai])

    def logreg(tri, vai):
        m = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=6)
        m.fit(Xs[tri], y[tri]); return m.predict_proba(Xs[vai])

    def knn(tri, vai):
        rs = np.random.RandomState(SEED)
        samp = np.concatenate([rs.choice(tri[y[tri] == c], min(15000, (y[tri] == c).sum()), replace=False)
                               for c in range(3)])
        m = KNeighborsClassifier(n_neighbors=80, weights="distance", n_jobs=6)
        m.fit(Xs[samp], y[samp]); return m.predict_proba(Xs[vai])

    algos = [("LightGBM", lgbm), ("XGBoost", xgb), ("HistGradientBoosting", histgb),
             ("RandomForest", rf), ("ExtraTrees", et),
             ("LogisticRegression", logreg), ("KNN", knn)]
    print(f"Comparison on stratified subsample of {len(y):,} rows, 5-fold, balanced accuracy:\n")
    res = {}
    for name, fn in algos:
        t0 = time.time()
        try:
            res[name] = cv(fn, y)
            print(f"  {name:26s} {res[name]:.5f}   ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"  {name:26s} FAILED: {e}", flush=True)
    print("\n=== ranking ===")
    for name, ba in sorted(res.items(), key=lambda kv: -kv[1]):
        print(f"  {name:26s} {ba:.5f}")


if __name__ == "__main__":
    main()
