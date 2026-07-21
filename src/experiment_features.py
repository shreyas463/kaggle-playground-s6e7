"""Test whether missingness indicators (+ lean features) beat the 0.9496 plateau.
Single LightGBM, full data, 5-fold OOF, balanced-accuracy decision-rule tuned."""
import os, sys, warnings
os.environ.setdefault("OMP_NUM_THREADS", "7")
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSES = ["at-risk", "unhealthy", "fit"]; C2I = {c: i for i, c in enumerate(CLASSES)}; I2C = {i: c for c, i in C2I.items()}
NUM = ["sleep_duration", "heart_rate", "bmi", "calorie_expenditure", "step_count", "exercise_duration", "water_intake"]
CAT = ["diet_type", "stress_level", "sleep_quality", "physical_activity_level", "smoking_alcohol", "gender"]
ORD = {"stress_level": {"low": 0, "medium": 1, "high": 2}, "sleep_quality": {"poor": 0, "average": 1, "good": 2},
       "physical_activity_level": {"sedentary": 0, "moderate": 1, "active": 2}, "smoking_alcohol": {"no": 0, "occasional": 1, "yes": 2}}
SEED = 42


def feats(df, add_missing):
    df = df.copy()
    for c in CAT:
        df[c] = df[c].astype("category")
    cols = list(NUM) + list(CAT)
    for c, mp in ORD.items():
        df[c + "_ord"] = df[c].map(mp).astype("float32"); cols.append(c + "_ord")
    df["steps_per_cal"] = df["step_count"] / (df["calorie_expenditure"] + 1)
    df["cal_per_min_ex"] = df["calorie_expenditure"] / (df["exercise_duration"] + 1)
    df["activity_score"] = df["step_count"] / 1000 + df["exercise_duration"] / 10
    cols += ["steps_per_cal", "cal_per_min_ex", "activity_score"]
    if add_missing:
        for c in NUM + CAT:
            df["miss_" + c] = df[c].isna().astype("int8"); cols.append("miss_" + c)
        df["n_missing"] = df[[c for c in NUM + CAT]].isna().sum(axis=1); cols.append("n_missing")
    return df, cols


def bw(y):
    c = np.bincount(y, minlength=3); return (len(y) / (3 * c))[y]


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


def run(add_missing, label):
    tr = pd.read_csv(os.path.join(ROOT, "data", "train.csv"))
    te = pd.read_csv(os.path.join(ROOT, "data", "test.csv"))
    y = tr["health_condition"].map(C2I).values.astype(int)
    trf, cols = feats(tr, add_missing); tef, _ = feats(te, add_missing)
    cat_idx = [cols.index(c) for c in CAT]
    oof = np.zeros((len(trf), 3)); test = np.zeros((len(tef), 3))
    for tri, vai in StratifiedKFold(5, shuffle=True, random_state=SEED).split(trf, y):
        d = lgb.Dataset(trf[cols].iloc[tri], y[tri], weight=bw(y[tri]), categorical_feature=cat_idx)
        dv = lgb.Dataset(trf[cols].iloc[vai], y[vai], weight=bw(y[vai]), categorical_feature=cat_idx, reference=d)
        m = lgb.train(dict(objective="multiclass", num_class=3, metric="multi_logloss", learning_rate=0.03,
                           num_leaves=127, min_child_samples=200, feature_fraction=0.8, bagging_fraction=0.8,
                           bagging_freq=1, lambda_l1=1.0, lambda_l2=2.0, num_threads=7, verbose=-1, seed=SEED),
                      d, num_boost_round=5000, valid_sets=[dv], callbacks=[lgb.early_stopping(200, verbose=False)])
        oof[vai] = m.predict(trf[cols].iloc[vai], num_iteration=m.best_iteration)
        test += m.predict(tef[cols], num_iteration=m.best_iteration) / 5
    w, tuned = tune(oof, y)
    print(f"[{label}] OOF balanced acc: raw={balanced_accuracy_score(y, oof.argmax(1)):.5f} tuned={tuned:.5f}", flush=True)
    sub = pd.DataFrame({"id": te["id"], "health_condition": [I2C[i] for i in (test * w).argmax(1)]})
    sub.to_csv(os.path.join(ROOT, "submissions", f"sub_{label}.csv"), index=False)
    return tuned


if __name__ == "__main__":
    a = run(False, "lean")
    b = run(True, "lean_missing")
    print(f"\nlean={a:.5f}  lean+missing={b:.5f}  delta={b-a:+.5f}")
