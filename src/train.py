"""Playground S6E7 — Predicting Student Health Risk.

Balanced-accuracy 3-class classification with LightGBM:
  - native categorical + missing-value handling
  - class-balanced sample weights (each class contributes equally to the loss)
  - stratified K-fold OOF
  - decision rule (per-class probability multipliers) tuned on OOF to maximize
    balanced accuracy, then applied to averaged test-fold probabilities.
"""
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, classification_report

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
SUB = os.path.join(ROOT, "submissions")
OOF = os.path.join(ROOT, "oof")
os.makedirs(SUB, exist_ok=True)
os.makedirs(OOF, exist_ok=True)

CLASSES = ["at-risk", "unhealthy", "fit"]
C2I = {c: i for i, c in enumerate(CLASSES)}
I2C = {i: c for c, i in C2I.items()}

NUM = ["sleep_duration", "heart_rate", "bmi", "calorie_expenditure",
       "step_count", "exercise_duration", "water_intake"]
CAT = ["diet_type", "stress_level", "sleep_quality",
       "physical_activity_level", "smoking_alcohol", "gender"]
# genuine ordinals -> add integer-encoded copies (trees can use either)
ORD = {
    "stress_level": {"low": 0, "medium": 1, "high": 2},
    "sleep_quality": {"poor": 0, "average": 1, "good": 2},
    "physical_activity_level": {"sedentary": 0, "moderate": 1, "active": 2},
    "smoking_alcohol": {"no": 0, "occasional": 1, "yes": 2},
}

N_FOLDS = 5
SEED = 42

PARAMS = dict(
    objective="multiclass", num_class=3, metric="multi_logloss",
    learning_rate=0.03, num_leaves=127, min_child_samples=200,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
    lambda_l1=1.0, lambda_l2=2.0, max_depth=-1, num_threads=0,
    verbose=-1, seed=SEED,
)


def load():
    tr = pd.read_csv(os.path.join(DATA, "train.csv"))
    te = pd.read_csv(os.path.join(DATA, "test.csv"))
    return tr, te


def make_features(df):
    df = df.copy()
    for c in CAT:
        df[c] = df[c].astype("category")
    feats = list(NUM) + list(CAT)
    for c, mp in ORD.items():
        df[c + "_ord"] = df[c].map(mp).astype("float32")
        feats.append(c + "_ord")
    # a few cheap numeric interactions with real-world meaning
    df["steps_per_cal"] = df["step_count"] / (df["calorie_expenditure"] + 1)
    df["cal_per_min_ex"] = df["calorie_expenditure"] / (df["exercise_duration"] + 1)
    df["activity_score"] = df["step_count"] / 1000 + df["exercise_duration"] / 10
    for c in ["steps_per_cal", "cal_per_min_ex", "activity_score"]:
        feats.append(c)
    return df, feats


def balanced_weights(y):
    n = len(y)
    k = len(np.unique(y))
    counts = np.bincount(y, minlength=k)
    w_per_class = n / (k * counts)
    return w_per_class[y]


def tune_decision_rule(oof, y):
    """Coordinate-ascent over per-class probability multipliers to maximize
    balanced accuracy of argmax(oof * w)."""
    def score(w):
        return balanced_accuracy_score(y, np.argmax(oof * w, axis=1))
    w = np.ones(3)
    best = score(w)
    grid = np.concatenate([np.linspace(0.3, 3.0, 28)])
    for _ in range(6):
        improved = False
        for k in range(3):
            base = w.copy()
            bw, bs = w[k], best
            for m in grid:
                base[k] = m
                s = score(base)
                if s > bs:
                    bs, bw = s, m
            if bw != w[k]:
                w[k] = bw
                best = bs
                improved = True
        if not improved:
            break
    return w, best


def main():
    tr, te = load()
    y = tr["health_condition"].map(C2I).values.astype(int)
    trf, feats = make_features(tr)
    tef, _ = make_features(te)
    X = trf[feats]
    Xt = tef[feats]
    cat_idx = [feats.index(c) for c in CAT]

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(X), 3))
    test_pred = np.zeros((len(Xt), 3))
    fold_scores = []

    for fold, (tri, vai) in enumerate(skf.split(X, y)):
        Xtr, Xva = X.iloc[tri], X.iloc[vai]
        ytr, yva = y[tri], y[vai]
        wtr = balanced_weights(ytr)
        dtr = lgb.Dataset(Xtr, ytr, weight=wtr, categorical_feature=cat_idx)
        dva = lgb.Dataset(Xva, yva, weight=balanced_weights(yva),
                          categorical_feature=cat_idx, reference=dtr)
        model = lgb.train(
            PARAMS, dtr, num_boost_round=5000, valid_sets=[dva],
            callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)],
        )
        oof[vai] = model.predict(Xva, num_iteration=model.best_iteration)
        test_pred += model.predict(Xt, num_iteration=model.best_iteration) / N_FOLDS
        fs = balanced_accuracy_score(yva, np.argmax(oof[vai], axis=1))
        fold_scores.append(fs)
        print(f"  fold {fold}: best_iter={model.best_iteration}  balanced_acc(argmax)={fs:.5f}")

    raw = balanced_accuracy_score(y, np.argmax(oof, axis=1))
    w, tuned = tune_decision_rule(oof, y)
    print(f"\nOOF balanced accuracy: argmax={raw:.5f}  tuned={tuned:.5f}  (folds {np.mean(fold_scores):.5f}+/-{np.std(fold_scores):.5f})")
    print("tuned class multipliers:", np.round(w, 3))
    print("\nOOF report (tuned rule):")
    print(classification_report(y, np.argmax(oof * w, axis=1), target_names=CLASSES, digits=4))

    # feature importance snapshot from the last fold model
    imp = pd.Series(model.feature_importance(importance_type="gain"), index=feats).sort_values(ascending=False)
    print("top features (gain):", dict(imp.head(10).round(0)))

    # apply tuned rule to test
    test_labels = np.argmax(test_pred * w, axis=1)
    sub = pd.DataFrame({"id": te["id"], "health_condition": [I2C[i] for i in test_labels]})
    out = os.path.join(SUB, "sub_lgbm.csv")
    sub.to_csv(out, index=False)
    np.save(os.path.join(OOF, "oof_lgbm.npy"), oof)
    np.save(os.path.join(OOF, "test_lgbm.npy"), test_pred)
    print("\nwrote", out)
    print("submission class distribution:\n", sub["health_condition"].value_counts(normalize=True).round(4))


if __name__ == "__main__":
    main()
