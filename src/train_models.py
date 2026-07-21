"""Train several algorithms under identical CV and save OOF/test probabilities.
Models: LightGBM, XGBoost, CatBoost, HistGradientBoosting, RandomForest,
ExtraTrees, LogisticRegression, KNN. Balanced-accuracy metric throughout."""
import os, sys, time, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import build_matrices, C2I

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
ART = os.path.join(ROOT, "artifacts")
os.makedirs(ART, exist_ok=True)
SEED, N_FOLDS = 42, 5

ONLY = set(sys.argv[1:])  # optionally run a subset: python train_models.py lgbm xgb


def bw(y):
    n, k = len(y), len(np.unique(y))
    c = np.bincount(y, minlength=k)
    return (n / (k * c))[y]


def save(name, oof, test, y):
    np.save(os.path.join(ART, f"oof_{name}.npy"), oof)
    np.save(os.path.join(ART, f"test_{name}.npy"), test)
    ba = balanced_accuracy_score(y, np.argmax(oof, axis=1))
    print(f">>> {name:12s} OOF balanced_acc = {ba:.5f}", flush=True)
    return ba


def folds(y):
    return StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y)


# ---------------- gradient-boosted trees (native categorical + NaN) ----------------
def run_lgbm(Xn, Xtn, cat_idx, y):
    import lightgbm as lgb
    p = dict(objective="multiclass", num_class=3, metric="multi_logloss",
             learning_rate=0.03, num_leaves=127, min_child_samples=200,
             feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
             lambda_l1=1.0, lambda_l2=2.0, num_threads=0, verbose=-1, seed=SEED)
    oof = np.zeros((len(Xn), 3)); test = np.zeros((len(Xtn), 3))
    for tri, vai in folds(y):
        dtr = lgb.Dataset(Xn.iloc[tri], y[tri], weight=bw(y[tri]), categorical_feature=cat_idx)
        dva = lgb.Dataset(Xn.iloc[vai], y[vai], weight=bw(y[vai]), categorical_feature=cat_idx, reference=dtr)
        m = lgb.train(p, dtr, num_boost_round=5000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        oof[vai] = m.predict(Xn.iloc[vai], num_iteration=m.best_iteration)
        test += m.predict(Xtn, num_iteration=m.best_iteration) / N_FOLDS
    return oof, test


def run_xgb(Xn, Xtn, y):
    from xgboost import XGBClassifier
    oof = np.zeros((len(Xn), 3)); test = np.zeros((len(Xtn), 3))
    for tri, vai in folds(y):
        m = XGBClassifier(n_estimators=3000, learning_rate=0.03, max_depth=8,
                          subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                          tree_method="hist", enable_categorical=True,
                          early_stopping_rounds=100, eval_metric="mlogloss",
                          n_jobs=0, random_state=SEED)
        m.fit(Xn.iloc[tri], y[tri], sample_weight=bw(y[tri]),
              eval_set=[(Xn.iloc[vai], y[vai])], sample_weight_eval_set=[bw(y[vai])], verbose=False)
        oof[vai] = m.predict_proba(Xn.iloc[vai])
        test += m.predict_proba(Xtn) / N_FOLDS
    return oof, test


def run_catboost(Xn, Xtn, cat_cols, y):
    from catboost import CatBoostClassifier, Pool
    Xc = Xn.copy(); Xtc = Xtn.copy()
    for c in cat_cols:
        Xc[c] = Xc[c].astype(str).fillna("nan"); Xtc[c] = Xtc[c].astype(str).fillna("nan")
    oof = np.zeros((len(Xc), 3)); test = np.zeros((len(Xtc), 3))
    for tri, vai in folds(y):
        m = CatBoostClassifier(iterations=3000, learning_rate=0.03, depth=8, l2_leaf_reg=3.0,
                               loss_function="MultiClass", eval_metric="MultiClass",
                               random_seed=SEED, thread_count=-1, verbose=False,
                               early_stopping_rounds=100)
        tr_pool = Pool(Xc.iloc[tri], y[tri], cat_features=cat_cols, weight=bw(y[tri]))
        va_pool = Pool(Xc.iloc[vai], y[vai], cat_features=cat_cols, weight=bw(y[vai]))
        m.fit(tr_pool, eval_set=va_pool)
        oof[vai] = m.predict_proba(Xc.iloc[vai])
        test += m.predict_proba(Xtc) / N_FOLDS
    return oof, test


def run_histgb(Xn, Xtn, y):
    from sklearn.ensemble import HistGradientBoostingClassifier
    oof = np.zeros((len(Xn), 3)); test = np.zeros((len(Xtn), 3))
    for tri, vai in folds(y):
        m = HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=1000, max_leaf_nodes=63, l2_regularization=1.0,
            categorical_features="from_dtype", early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=50, class_weight="balanced", random_state=SEED)
        m.fit(Xn.iloc[tri], y[tri])
        oof[vai] = m.predict_proba(Xn.iloc[vai])
        test += m.predict_proba(Xtn) / N_FOLDS
    return oof, test


# ---------------- classic sklearn (encoded / scaled matrices) ----------------
def run_generic(factory, X, Xt, y, weighted=True):
    oof = np.zeros((X.shape[0], 3)); test = np.zeros((Xt.shape[0], 3))
    Xa = X.values if hasattr(X, "values") else X
    for tri, vai in folds(y):
        m = factory()
        if weighted:
            m.fit(Xa[tri], y[tri], sample_weight=bw(y[tri]))
        else:
            m.fit(Xa[tri], y[tri])
        oof[vai] = m.predict_proba(Xa[vai])
        test += m.predict_proba(Xt if not hasattr(Xt, "values") else Xt.values) / N_FOLDS
    return oof, test


def run_knn(Xs, Xts, y):
    from sklearn.neighbors import KNeighborsClassifier
    # subsample train for tractability; class-balanced sample to help balanced-acc
    rng = np.random.RandomState(SEED)
    idx_by_c = [np.where(y == c)[0] for c in range(3)]
    per = 40000
    samp = np.concatenate([rng.choice(ix, min(per, len(ix)), replace=False) for ix in idx_by_c])
    oof = np.zeros((Xs.shape[0], 3)); test = np.zeros((Xts.shape[0], 3))
    for tri, vai in folds(y):
        tr_samp = np.intersect1d(tri, samp)
        m = KNeighborsClassifier(n_neighbors=100, weights="distance", n_jobs=-1)
        m.fit(Xs[tr_samp], y[tr_samp])
        oof[vai] = m.predict_proba(Xs[vai])
        test += m.predict_proba(Xts) / N_FOLDS
    return oof, test


def main():
    tr = pd.read_csv(os.path.join(DATA, "train.csv"))
    te = pd.read_csv(os.path.join(DATA, "test.csv"))
    y = tr["health_condition"].map(C2I).values.astype(int)
    M = build_matrices(tr, te)
    Xn, Xtn, cat_idx, cat_cols, num_cols = M["native"]
    Xe, Xte = M["enc"]; Xs, Xts = M["scaled"]

    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
    from sklearn.linear_model import LogisticRegression

    jobs = [
        ("lgbm",       lambda: run_lgbm(Xn, Xtn, cat_idx, y)),
        ("xgb",        lambda: run_xgb(Xn, Xtn, y)),
        ("catboost",   lambda: run_catboost(Xn, Xtn, cat_cols, y)),
        ("histgb",     lambda: run_histgb(Xn, Xtn, y)),
        ("rf",         lambda: run_generic(lambda: RandomForestClassifier(
                            n_estimators=400, max_depth=18, min_samples_leaf=20,
                            class_weight="balanced", n_jobs=-1, random_state=SEED), Xe, Xte, y, weighted=False)),
        ("extratrees", lambda: run_generic(lambda: ExtraTreesClassifier(
                            n_estimators=500, max_depth=22, min_samples_leaf=15,
                            class_weight="balanced", n_jobs=-1, random_state=SEED), Xe, Xte, y, weighted=False)),
        ("logreg",     lambda: run_generic(lambda: LogisticRegression(
                            max_iter=2000, C=1.0, class_weight="balanced", n_jobs=-1), Xs, Xts, y, weighted=False)),
        ("knn",        lambda: run_knn(Xs, Xts, y)),
    ]
    results = {}
    for name, fn in jobs:
        if ONLY and name not in ONLY:
            continue
        if os.path.exists(os.path.join(ART, f"oof_{name}.npy")):
            oof = np.load(os.path.join(ART, f"oof_{name}.npy"))
            results[name] = balanced_accuracy_score(y, np.argmax(oof, axis=1))
            print(f"--- {name}: cached ({results[name]:.5f})", flush=True)
            continue
        t0 = time.time()
        try:
            oof, test = fn()
            results[name] = save(name, oof, test, y)
            print(f"    ({name} took {time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            import traceback
            print(f"!!! {name} FAILED: {e}", flush=True)
            traceback.print_exc()
    print("\n=== SUMMARY (OOF balanced accuracy) ===")
    for name, ba in sorted(results.items(), key=lambda kv: -kv[1]):
        print(f"  {name:12s} {ba:.5f}")


if __name__ == "__main__":
    main()
