"""Bayes-rule model: exploit the known deterministic generation rule.

The label is (verified) a deterministic tree on 3 drivers:
    sleep<6 : stress==high -> unhealthy ; else -> at-risk
    sleep>=6: stress==low AND activity==active AND sleep>=7 -> fit ; else -> at-risk

For each row we build a distribution over each driver:
  - observed  -> a point mass (the noisy observed value)
  - missing   -> an imputed distribution predicted from all other features (OOF, no leak)
and marginalize the rule to get P(label | observed features):
    P(unhealthy) = P(sleep<6) * P(stress=high)
    P(fit)       = P(sleep>=7) * P(stress=low) * P(activity=active)
    P(at-risk)   = 1 - P(unhealthy) - P(fit)

Outputs OOF + test probabilities (oof_rulebayes.npy / test_rulebayes.npy) for the blend.
"""
import os, warnings
os.environ.setdefault("OMP_NUM_THREADS", "7")
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSES = ["at-risk", "unhealthy", "fit"]; C2I = {c: i for i, c in enumerate(CLASSES)}; I2C = {i: c for c, i in C2I.items()}
NUM = ["sleep_duration", "heart_rate", "bmi", "calorie_expenditure", "step_count", "exercise_duration", "water_intake"]
CAT = ["diet_type", "stress_level", "sleep_quality", "physical_activity_level", "smoking_alcohol", "gender"]
SEED = 42
LGP = dict(n_estimators=400, num_leaves=63, learning_rate=0.05, min_child_samples=100,
           subsample=0.8, colsample_bytree=0.8, verbose=-1, n_jobs=7, random_state=SEED)


def design(df, exclude):
    cols = [c for c in NUM + CAT if c not in exclude]
    X = df[cols].copy()
    for c in cols:
        if c in CAT:
            X[c] = X[c].astype("category")
    return X


def oof_predict(Xtr, ytr, Xte, folds, n_class, mask=None):
    """5-fold OOF proba on Xtr (rows in `mask` used for training) + averaged test proba."""
    oof = np.full((len(Xtr), n_class), np.nan)
    test = np.zeros((len(Xte), n_class))
    for tri, vai in folds:
        tr_idx = tri if mask is None else tri[mask[tri]]
        m = lgb.LGBMClassifier(**LGP, objective="multiclass" if n_class > 2 else "binary")
        m.fit(Xtr.iloc[tr_idx], ytr[tr_idx])
        p = m.predict_proba(Xtr.iloc[vai])
        pt = m.predict_proba(Xte)
        if n_class == 2:
            oof[vai, 1] = p[:, 1]; oof[vai, 0] = p[:, 0]
            test[:, 1] += pt[:, 1] / len(folds); test[:, 0] += pt[:, 0] / len(folds)
        else:
            oof[vai] = p; test += pt / len(folds)
    return oof, test


def driver_dists(tr, te, folds):
    """Return per-row P(stress=low/high), P(activity=active), P(sleep<6), P(sleep<7)
    for both train (OOF) and test, honoring observed values as point masses."""
    n_tr, n_te = len(tr), len(te)
    out = {}

    # stress_level (low/med/high) -> need P(low), P(high)
    s_map = {"low": 0, "medium": 1, "high": 2}
    obs = tr["stress_level"].notna().values
    ys = tr["stress_level"].map(s_map).fillna(0).astype(int).values
    Xs_tr, Xs_te = design(tr, ["stress_level"]), design(te, ["stress_level"])
    oof_s, test_s = oof_predict(Xs_tr, ys, Xs_te, folds, 3, mask=obs)
    # observed rows -> point mass
    for i, cls in [(0, "low"), (2, "high")]:
        col = f"stress_{cls}"
        tr_p = oof_s[:, i].copy(); te_p = test_s[:, i].copy()
        known = tr["stress_level"].values == cls
        tr_p = np.where(tr["stress_level"].notna().values, (known).astype(float), tr_p)
        te_known = te["stress_level"].values == cls
        te_p = np.where(te["stress_level"].notna().values, te_known.astype(float), te_p)
        out[col] = (tr_p, te_p)

    # physical_activity_level -> P(active)
    a_map = {"sedentary": 0, "moderate": 1, "active": 2}
    obs = tr["physical_activity_level"].notna().values
    ya = tr["physical_activity_level"].map(a_map).fillna(0).astype(int).values
    Xa_tr, Xa_te = design(tr, ["physical_activity_level"]), design(te, ["physical_activity_level"])
    oof_a, test_a = oof_predict(Xa_tr, ya, Xa_te, folds, 3, mask=obs)
    tr_p = np.where(tr["physical_activity_level"].notna().values,
                    (tr["physical_activity_level"].values == "active").astype(float), oof_a[:, 2])
    te_p = np.where(te["physical_activity_level"].notna().values,
                    (te["physical_activity_level"].values == "active").astype(float), test_a[:, 2])
    out["act_active"] = (tr_p, te_p)

    # sleep thresholds: observed sleep -> known bin; missing -> predict from all-except-sleep
    for thr, key in [(6, "sleep_lt6"), (7, "sleep_lt7")]:
        obs = tr["sleep_duration"].notna().values
        yb = (tr["sleep_duration"] < thr).fillna(False).astype(int).values
        Xz_tr, Xz_te = design(tr, ["sleep_duration"]), design(te, ["sleep_duration"])
        oof_z, test_z = oof_predict(Xz_tr, yb, Xz_te, folds, 2, mask=obs)
        tr_p = np.where(tr["sleep_duration"].notna().values,
                        (tr["sleep_duration"].values < thr).astype(float), oof_z[:, 1])
        te_p = np.where(te["sleep_duration"].notna().values,
                        (te["sleep_duration"].values < thr).astype(float), test_z[:, 1])
        out[key] = (tr_p, te_p)
    return out


def apply_rule(d, which):
    """Marginalize the generation rule -> P(label). `which` selects train(0)/test(1)."""
    p_lt6 = np.clip(d["sleep_lt6"][which], 0, 1)
    p_lt7 = np.clip(d["sleep_lt7"][which], 0, 1)
    p_ge7 = np.clip(1 - p_lt7, 0, 1)
    p_high = np.clip(d["stress_high"][which], 0, 1)
    p_low = np.clip(d["stress_low"][which], 0, 1)
    p_act = np.clip(d["act_active"][which], 0, 1)
    P = np.zeros((len(p_lt6), 3))
    P[:, C2I["unhealthy"]] = p_lt6 * p_high
    P[:, C2I["fit"]] = p_ge7 * p_low * p_act
    P[:, C2I["at-risk"]] = np.clip(1 - P[:, C2I["unhealthy"]] - P[:, C2I["fit"]], 0, 1)
    P /= P.sum(1, keepdims=True) + 1e-9
    return P


def tune(prob, y):
    best, w = balanced_accuracy_score(y, prob.argmax(1)), np.ones(3)
    for _ in range(6):
        imp = False
        for k in range(3):
            b = w.copy(); bw, bs = w[k], best
            for m in np.linspace(0.4, 2.5, 30):
                b[k] = m; s = balanced_accuracy_score(y, (prob * b).argmax(1))
                if s > bs: bs, bw = s, m
            if bw != w[k]: w[k], best, imp = bw, bs, True
        if not imp: break
    return w, best


def main():
    tr = pd.read_csv(os.path.join(ROOT, "data", "train.csv"))
    te = pd.read_csv(os.path.join(ROOT, "data", "test.csv"))
    y = tr["health_condition"].map(C2I).values.astype(int)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(tr, y))
    d = driver_dists(tr, te, folds)
    oof = apply_rule(d, 0); test = apply_rule(d, 1)
    w, tuned = tune(oof, y)
    print(f"rule-bayes OOF balanced acc: raw={balanced_accuracy_score(y, oof.argmax(1)):.5f} tuned={tuned:.5f}", flush=True)
    np.save(os.path.join(ROOT, "artifacts", "oof_rulebayes.npy"), oof)
    np.save(os.path.join(ROOT, "artifacts", "test_rulebayes.npy"), test)
    sub = pd.DataFrame({"id": te["id"], "health_condition": [I2C[i] for i in (test * w).argmax(1)]})
    sub.to_csv(os.path.join(ROOT, "submissions", "sub_rulebayes.csv"), index=False)
    print("saved oof_rulebayes / test_rulebayes / sub_rulebayes.csv", flush=True)


if __name__ == "__main__":
    main()
