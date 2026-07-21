"""Shared feature engineering for S6E7."""
import numpy as np
import pandas as pd

CLASSES = ["at-risk", "unhealthy", "fit"]
C2I = {c: i for i, c in enumerate(CLASSES)}
I2C = {i: c for c, i in C2I.items()}

NUM = ["sleep_duration", "heart_rate", "bmi", "calorie_expenditure",
       "step_count", "exercise_duration", "water_intake"]
CAT = ["diet_type", "stress_level", "sleep_quality",
       "physical_activity_level", "smoking_alcohol", "gender"]
ORD = {
    "stress_level": {"low": 0, "medium": 1, "high": 2},
    "sleep_quality": {"poor": 0, "average": 1, "good": 2},
    "physical_activity_level": {"sedentary": 0, "moderate": 1, "active": 2},
    "smoking_alcohol": {"no": 0, "occasional": 1, "yes": 2},
}


def engineer(df):
    """Return (df_with_features, categorical_cols, numeric_cols)."""
    df = df.copy()
    # ordinal encodings (genuine orderings)
    for c, mp in ORD.items():
        df[c + "_ord"] = df[c].map(mp).astype("float32")
    # interaction categoricals (capture stress x activity -> fit, etc.)
    df["stress_activity"] = (df["stress_level"].astype(str) + "_" +
                             df["physical_activity_level"].astype(str))
    df["stress_sleepq"] = (df["stress_level"].astype(str) + "_" +
                           df["sleep_quality"].astype(str))
    # numeric interactions / ratios
    df["steps_per_cal"] = df["step_count"] / (df["calorie_expenditure"] + 1)
    df["cal_per_min_ex"] = df["calorie_expenditure"] / (df["exercise_duration"] + 1)
    df["activity_score"] = df["step_count"] / 1000 + df["exercise_duration"] / 10
    df["sleep_x_stress"] = df["sleep_duration"] * df["stress_level_ord"]
    df["sleep_x_activity"] = df["sleep_duration"] * df["physical_activity_level_ord"]

    cat_cols = CAT + ["stress_activity", "stress_sleepq"]
    num_cols = NUM + [c + "_ord" for c in ORD] + [
        "steps_per_cal", "cal_per_min_ex", "activity_score",
        "sleep_x_stress", "sleep_x_activity"]
    return df, cat_cols, num_cols


def build_matrices(tr, te):
    """Three design matrices for the different model families."""
    trf, cat_cols, num_cols = engineer(tr)
    tef, _, _ = engineer(te)

    # 1) native: category dtype + NaN (LGBM / XGB / CatBoost / HistGB)
    Xn = trf[num_cols + cat_cols].copy()
    Xtn = tef[num_cols + cat_cols].copy()
    for c in cat_cols:
        Xn[c] = Xn[c].astype("category")
        Xtn[c] = Xtn[c].astype("category")

    # 2) encoded: median-impute numeric, ordinal-encode cats (RF / ExtraTrees)
    from sklearn.preprocessing import OrdinalEncoder
    Xe = trf[num_cols + cat_cols].copy()
    Xte = tef[num_cols + cat_cols].copy()
    med = Xe[num_cols].median()
    Xe[num_cols] = Xe[num_cols].fillna(med)
    Xte[num_cols] = Xte[num_cols].fillna(med)
    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                        encoded_missing_value=-2)
    Xe[cat_cols] = oe.fit_transform(Xe[cat_cols].astype(str))
    Xte[cat_cols] = oe.transform(Xte[cat_cols].astype(str))

    # 3) scaled: impute + one-hot + standardize (LogReg / KNN)
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from scipy.sparse import hstack, csr_matrix
    sc = StandardScaler()
    Xs_num = sc.fit_transform(Xe[num_cols])
    Xts_num = sc.transform(Xte[num_cols])
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    Xs_cat = ohe.fit_transform(trf[cat_cols].astype(str).fillna("nan"))
    Xts_cat = ohe.transform(tef[cat_cols].astype(str).fillna("nan"))
    Xs = hstack([csr_matrix(Xs_num), Xs_cat]).tocsr()
    Xts = hstack([csr_matrix(Xts_num), Xts_cat]).tocsr()

    cat_idx = [list(Xn.columns).index(c) for c in cat_cols]
    return dict(native=(Xn, Xtn, cat_idx, cat_cols, num_cols),
                enc=(Xe, Xte), scaled=(Xs, Xts))
