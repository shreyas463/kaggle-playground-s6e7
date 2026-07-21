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

# --- 1. Ensure torch has kernels for the assigned GPU BEFORE importing torch ---
# Kaggle's torch (2.10+cu128) dropped Pascal (sm_60) kernels, but it assigns P100 GPUs.
# Probe in a subprocess (so the main process's torch import stays unlocked), and if the
# GPU arch isn't supported, pin torch 2.5.1+cu121 (has sm_60 and satisfies tabpfn>=2.5).
_probe = subprocess.run(
    [sys.executable, "-c",
     "import torch;"
     "cap=torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0,0);"
     "print('OK' if f'sm_{cap[0]}{cap[1]}' in torch.cuda.get_arch_list() else 'PIN')"],
    capture_output=True, text=True)
if "PIN" in _probe.stdout:
    print("GPU arch unsupported by current torch -> pinning torch 2.5.1+cu121", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "torch==2.5.1", "--index-url", "https://download.pytorch.org/whl/cu121"], check=False)

# --- 2. install tabpfn WITHOUT letting pip replace the (now-pinned) torch ---
try:
    import tabpfn  # noqa
except Exception:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "tabpfn"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "safetensors>=0.4.0", "einops>=0.4.0", "pydantic>=2.8.0",
                    "pydantic-settings>=2.10.1", "huggingface-hub>=0.23.0"], check=False)

import glob
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
import torch


def find_data_dir():
    root = "/kaggle/input"
    print("::/kaggle/input contents::", os.listdir(root) if os.path.isdir(root) else "MISSING", flush=True)
    hits = glob.glob(f"{root}/**/train.csv", recursive=True)
    print("::train.csv candidates::", hits, flush=True)
    if hits:
        return os.path.dirname(hits[0])
    raise FileNotFoundError("competition data not mounted under /kaggle/input")


def cuda_report():
    print("torch", torch.__version__, "| cuda", torch.version.cuda,
          "| avail", torch.cuda.is_available(), flush=True)
    if torch.cuda.is_available():
        try:
            print("gpu", torch.cuda.get_device_name(0),
                  "| capability", torch.cuda.get_device_capability(0),
                  "| arch_list", torch.cuda.get_arch_list(), flush=True)
            _ = (torch.randn(64, 64, device="cuda") @ torch.randn(64, 64, device="cuda")).sum().item()
            print("CUDA self-test: OK", flush=True)
            return True
        except Exception as e:
            print("CUDA self-test FAILED:", str(e)[:160], flush=True)
            return False
    return False


CUDA_OK = cuda_report()
DATA = find_data_dir()
OUT = "/kaggle/working"
CLASSES = ["at-risk", "unhealthy", "fit"]
C2I = {c: i for i, c in enumerate(CLASSES)}
SEED = 42
N_ENS = int(os.environ.get("N_ENS", "6"))          # subsample models per fit
CTX = int(os.environ.get("CTX", "8000"))           # rows per subsample (balanced)
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


# Force the UNGATED v2 classifier (v3/v2.5/v2.6 require a license; v2 does not).
V2_CKPT = os.environ.get("TABPFN_V2_CKPT", "tabpfn-v2-classifier-finetuned-zk73skhh.ckpt")


def make_clf():
    from tabpfn import TabPFNClassifier
    for kw in (dict(model_path=V2_CKPT, device=DEVICE, ignore_pretraining_limits=True),
               dict(model_path=V2_CKPT, device=DEVICE),
               dict(device=DEVICE, ignore_pretraining_limits=True)):
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
