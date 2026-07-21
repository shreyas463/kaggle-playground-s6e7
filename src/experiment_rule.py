"""Test the generation-model discoveries (discussion #717222):
the original target is a deterministic depth-4 tree over sleep_duration,
stress_level, physical_activity_level; original dataset = ziya07/college-student-
health-behavior-dataset (50k rows, no missing).

Configs (full data, LightGBM, 5-fold, balanced weights, tuned decision rule):
  A lean            : reference (~0.94966)
  B lean+rule       : + exact-threshold flags and deterministic rule label
  C lean+orig       : + original 50k appended to TRAIN folds only
  D lean+rule+orig  : both
"""
import os, sys, warnings
os.environ.setdefault("OMP_NUM_THREADS", "7")
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSES = ["at-risk", "unhealthy", "fit"]
C2I = {c: i for i, c in enumerate(CLASSES)}; I2C = {i: c for c, i in C2I.items()}
NUM = ["sleep_duration", "heart_rate", "bmi", "calorie_expenditure", "step_count", "exercise_duration", "water_intake"]
CAT = ["diet_type", "stress_level", "sleep_quality", "physical_activity_level", "smoking_alcohol", "gender"]
ORD = {"stress_level": {"low": 0, "medium": 1, "high": 2}, "sleep_quality": {"poor": 0, "average": 1, "good": 2},
       "physical_activity_level": {"sedentary": 0, "moderate": 1, "active": 2}, "smoking_alcohol": {"no": 0, "occasional": 1, "yes": 2}}
SEED = 42
PARAMS = dict(objective="multiclass", num_class=3, metric="multi_logloss", learning_rate=0.03,
              num_leaves=127, min_child_samples=200, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, lambda_l1=1.0, lambda_l2=2.0, num_threads=7, verbose=-1, seed=SEED)


def rule_label(df):
    """Deterministic generation rule (verified accuracy 1.0 on the original).
    Returns float codes with NaN where any needed feature is missing."""
    sd = df["sleep_duration"]; sl = df["stress_level"].astype(str); pa = df["physical_activity_level"].astype(str)
    out = np.select(
        [ (sd < 6) & (sl == "high"),
          (sd < 6),
          (sl == "low") & (pa == "active") & (sd >= 7) ],
        [ C2I["unhealthy"], C2I["at-risk"], C2I["fit"] ],
        default=C2I["at-risk"]).astype(float)
    miss = df["sleep_duration"].isna() | df["stress_level"].isna() | df["physical_activity_level"].isna()
    out[miss.values] = np.nan
    return out


def feats(df, add_rule):
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
    if add_rule:
        df["sleep_lt6"] = (df["sleep_duration"] < 6).astype("float32").where(df["sleep_duration"].notna())
        df["sleep_lt7"] = (df["sleep_duration"] < 7).astype("float32").where(df["sleep_duration"].notna())
        df["stress_low"] = (df["stress_level"].astype(str) == "low").astype("float32").where(df["stress_level"].notna())
        df["stress_high"] = (df["stress_level"].astype(str) == "high").astype("float32").where(df["stress_level"].notna())
        df["pa_active"] = (df["physical_activity_level"].astype(str) == "active").astype("float32").where(df["physical_activity_level"].notna())
        df["rule_label"] = rule_label(df)
        # distance to the two critical sleep thresholds (noise lives near boundaries)
        df["sleep_m6"] = df["sleep_duration"] - 6.0
        df["sleep_m7"] = df["sleep_duration"] - 7.0
        cols += ["sleep_lt6", "sleep_lt7", "stress_low", "stress_high", "pa_active",
                 "rule_label", "sleep_m6", "sleep_m7"]
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


def run(label, add_rule, add_orig, tr, te, orig):
    y = tr["health_condition"].map(C2I).values.astype(int)
    trf, cols = feats(tr, add_rule)
    tef, _ = feats(te, add_rule)
    ofr = None
    if add_orig:
        ofr, _ = feats(orig, add_rule)
        yo = orig["health_condition"].map(C2I).values.astype(int)
    cat_idx = [cols.index(c) for c in CAT]
    oof = np.zeros((len(trf), 3)); test = np.zeros((len(tef), 3))
    for tri, vai in StratifiedKFold(5, shuffle=True, random_state=SEED).split(trf, y):
        Xtr, ytr = trf[cols].iloc[tri], y[tri]
        if add_orig:  # append original to TRAIN portion only
            Xtr = pd.concat([Xtr, ofr[cols]], ignore_index=True)
            ytr = np.concatenate([ytr, yo])
        d = lgb.Dataset(Xtr, ytr, weight=bw(ytr), categorical_feature=cat_idx)
        dv = lgb.Dataset(trf[cols].iloc[vai], y[vai], weight=bw(y[vai]), categorical_feature=cat_idx, reference=d)
        m = lgb.train(PARAMS, d, num_boost_round=5000, valid_sets=[dv],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        oof[vai] = m.predict(trf[cols].iloc[vai], num_iteration=m.best_iteration)
        test += m.predict(tef[cols], num_iteration=m.best_iteration) / 5
    w, tuned = tune(oof, y)
    raw = balanced_accuracy_score(y, oof.argmax(1))
    print(f"[{label}] raw={raw:.5f} tuned={tuned:.5f}", flush=True)
    np.save(os.path.join(ROOT, "artifacts", f"oof_{label}.npy"), oof)
    np.save(os.path.join(ROOT, "artifacts", f"test_{label}.npy"), test)
    sub = pd.DataFrame({"id": te["id"], "health_condition": [I2C[i] for i in (test * w).argmax(1)]})
    sub.to_csv(os.path.join(ROOT, "submissions", f"sub_{label}.csv"), index=False)
    return tuned


if __name__ == "__main__":
    tr = pd.read_csv(os.path.join(ROOT, "data", "train.csv"))
    te = pd.read_csv(os.path.join(ROOT, "data", "test.csv"))
    orig = pd.read_csv(os.path.join(ROOT, "data", "orig", "ziya07_college", "student_health_dataset_50k.csv"))
    orig = orig[[c for c in tr.columns if c != "id"]]
    res = {}
    res["B_rule"] = run("B_rule", True, False, tr, te, orig)
    res["C_orig"] = run("C_orig", False, True, tr, te, orig)
    res["D_rule_orig"] = run("D_rule_orig", True, True, tr, te, orig)
    print("\nReference A_lean = 0.94966")
    for k, v in res.items():
        print(f"  {k:12s} {v:.5f}  ({v-0.94966:+.5f})")
