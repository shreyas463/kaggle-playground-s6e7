"""S6E7 TabPFN probabilities (Kaggle GPU kernel).

TabPFN is an in-context tabular foundation model whose inductive bias is orthogonal to the GBDT
ecosystem — the one thing shown to carry incremental signal past the ~0.950 Bayes frontier on this
dataset. Native context limit is ~10k rows, so we use a *subsample ensemble*: fit N_ENS models each
on a class-balanced ~10k subsample and average their probabilities. This is also where the diversity
comes from.

Outputs (kernel /kaggle/working):
  oof_tabpfn.npy   (n_train, 3)   out-of-fold probabilities for honest blending
  test_tabpfn.npy  (n_test, 3)    test probabilities
  meta.json        run info + OOF balanced accuracy
Classes are ordered ['at-risk','unhealthy','fit'] to match the local pipeline.
"""
import os, sys, json, time, subprocess, warnings
warnings.filterwarnings("ignore")

# --- install tabpfn (internet must be enabled on the kernel) ---
try:
    import tabpfn  # noqa
except Exception:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tabpfn"], check=False)

import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
import torch

DATA = "/kaggle/input/playground-series-s6e7"
OUT = "/kaggle/working"
CLASSES = ["at-risk", "unhealthy", "fit"]
C2I = {c: i for i, c in enumerate(CLASSES)}
SEED = 42
N_ENS = int(os.environ.get("N_ENS", "8"))          # subsample models per fit
CTX = int(os.environ.get("CTX", "10000"))          # rows per subsample (balanced)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM = ["sleep_duration", "heart_rate", "bmi", "calorie_expenditure", "step_count", "exercise_duration", "water_intake"]
CAT = ["diet_type", "stress_level", "sleep_quality", "physical_activity_level", "smoking_alcohol", "gender"]
ORD = {"stress_level": {"low": 0, "medium": 1, "high": 2}, "sleep_quality": {"poor": 0, "average": 1, "good": 2},
       "physical_activity_level": {"sedentary": 0, "moderate": 1, "active": 2}, "smoking_alcohol": {"no": 0, "occasional": 1, "yes": 2}}


def prep(df):
    """Numeric matrix for TabPFN: numerics + ordinal-encoded cats + rule-threshold flags.
    Missing values left as NaN (TabPFN handles them)."""
    X = df[NUM].copy()
    for c, mp in ORD.items():
        X[c + "_ord"] = df[c].map(mp)
    # nominal cats -> integer codes (unknown/NaN -> -1)
    for c in ["diet_type", "gender"]:
        X[c + "_code"] = df[c].astype("category").cat.codes.replace(-1, np.nan)
    # generation-rule threshold flags (the 3 true drivers)
    X["sleep_lt6"] = (df["sleep_duration"] < 6).astype(float).where(df["sleep_duration"].notna())
    X["sleep_lt7"] = (df["sleep_duration"] < 7).astype(float).where(df["sleep_duration"].notna())
    return X.astype("float32")


def make_clf():
    from tabpfn import TabPFNClassifier
    for kw in (dict(device=DEVICE, ignore_pretraining_limits=True),
               dict(device=DEVICE), dict()):
        try:
            return TabPFNClassifier(**kw)
        except TypeError:
            continue
    from tabpfn import TabPFNClassifier as T
    return T()


def balanced_subsample(y_pool, rng, per):
    idx = []
    for c in range(3):
        ci = np.where(y_pool == c)[0]
        idx.append(rng.choice(ci, min(per, len(ci)), replace=len(ci) < per))
    out = np.concatenate(idx); rng.shuffle(out)
    return out


def ensemble_predict(Xtr, ytr, Xpred):
    """Average N_ENS TabPFN models fit on balanced subsamples of (Xtr, ytr)."""
    per = max(1, CTX // 3)
    prob = np.zeros((len(Xpred), 3))
    Xtr_v, Xpred_v = Xtr.values, Xpred.values
    for e in range(N_ENS):
        rng = np.random.RandomState(SEED + e)
        s = balanced_subsample(ytr, rng, per)
        clf = make_clf()
        clf.fit(Xtr_v[s], ytr[s])
        # batch prediction to avoid GPU OOM on large query sets
        BATCH = 40000
        p = np.zeros((len(Xpred_v), 3))
        cls = None
        for b in range(0, len(Xpred_v), BATCH):
            pb = clf.predict_proba(Xpred_v[b:b + BATCH])
            if cls is None:
                cls = list(getattr(clf, "classes_", range(3)))
            for j, c in enumerate(cls):
                p[b:b + BATCH, int(c)] = pb[:, j]
        prob += p / N_ENS
        print(f"    ens {e+1}/{N_ENS} done", flush=True)
    return prob


def main():
    t0 = time.time()
    tr = pd.read_csv(f"{DATA}/train.csv")
    te = pd.read_csv(f"{DATA}/test.csv")
    y = tr["health_condition"].map(C2I).values.astype(int)
    Xtr, Xte = prep(tr), prep(te)
    print(f"device={DEVICE} N_ENS={N_ENS} CTX={CTX} train={Xtr.shape} test={Xte.shape}", flush=True)

    oof = np.zeros((len(Xtr), 3))
    for f, (tri, vai) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xtr, y)):
        print(f"fold {f} ...", flush=True)
        oof[vai] = ensemble_predict(Xtr.iloc[tri], y[tri], Xtr.iloc[vai])
    ba = balanced_accuracy_score(y, oof.argmax(1))
    print(f"OOF balanced accuracy = {ba:.5f}", flush=True)

    print("test predictions ...", flush=True)
    test = ensemble_predict(Xtr, y, Xte)

    np.save(f"{OUT}/oof_tabpfn.npy", oof)
    np.save(f"{OUT}/test_tabpfn.npy", test)
    # also a standalone submission from TabPFN alone (prior-corrected argmax)
    prior = np.bincount(y) / len(y)
    lab = (test / prior).argmax(1)
    pd.DataFrame({"id": te["id"], "health_condition": [CLASSES[i] for i in lab]}).to_csv(f"{OUT}/sub_tabpfn.csv", index=False)
    json.dump({"oof_balanced_accuracy": float(ba), "n_ens": N_ENS, "ctx": CTX,
               "device": DEVICE, "minutes": (time.time() - t0) / 60}, open(f"{OUT}/meta.json", "w"), indent=2)
    print(f"done in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
