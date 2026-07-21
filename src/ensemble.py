"""Greedy ensemble over saved model OOF/test probabilities + balanced-accuracy
decision-rule tuning -> submission."""
import os, sys, glob
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, classification_report

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import C2I, I2C, CLASSES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(ROOT, "artifacts")
SUB = os.path.join(ROOT, "submissions")
os.makedirs(SUB, exist_ok=True)


def tune_rule(prob, y):
    def score(w):
        return balanced_accuracy_score(y, np.argmax(prob * w, axis=1))
    w = np.ones(3); best = score(w)
    grid = np.linspace(0.3, 3.0, 55)
    for _ in range(8):
        improved = False
        for k in range(3):
            b = w.copy(); bw, bs = w[k], best
            for m in grid:
                b[k] = m; s = score(b)
                if s > bs:
                    bs, bw = s, m
            if bw != w[k]:
                w[k] = bw; best = bs; improved = True
        if not improved:
            break
    return w, best


def main():
    y = pd.read_csv(os.path.join(ROOT, "data", "train.csv"))["health_condition"].map(C2I).values.astype(int)
    names = sorted(os.path.basename(f)[4:-4] for f in glob.glob(os.path.join(ART, "oof_*.npy")))
    oofs = {n: np.load(os.path.join(ART, f"oof_{n}.npy")) for n in names}
    tests = {n: np.load(os.path.join(ART, f"test_{n}.npy")) for n in names}

    print("=== individual models (tuned balanced acc) ===")
    indiv = {}
    for n in names:
        _, ba = tune_rule(oofs[n], y)
        indiv[n] = ba
        print(f"  {n:12s} raw={balanced_accuracy_score(y, oofs[n].argmax(1)):.5f}  tuned={ba:.5f}")

    # greedy forward selection by tuned balanced accuracy (simple averaging)
    order = sorted(names, key=lambda n: -indiv[n])
    selected = [order[0]]
    cur = oofs[order[0]].copy()
    _, best = tune_rule(cur, y)
    improved = True
    while improved:
        improved = False
        for n in order:
            if n in selected:
                continue
            trial = (cur * len(selected) + oofs[n]) / (len(selected) + 1)
            _, ba = tune_rule(trial, y)
            if ba > best + 1e-6:
                best = ba; selected.append(n); cur = trial; improved = True
                print(f"  + added {n}: ensemble tuned balanced acc -> {best:.5f}")
                break
    print(f"\nselected ensemble: {selected}  OOF tuned balanced acc = {best:.5f}")

    # build test blend the same way, tune final rule on OOF, apply to test
    test_blend = np.mean([tests[n] for n in selected], axis=0)
    oof_blend = np.mean([oofs[n] for n in selected], axis=0)
    w, final = tune_rule(oof_blend, y)
    print("final decision multipliers:", np.round(w, 3), " OOF:", round(final, 5))
    print("\nOOF report (ensemble, tuned rule):")
    print(classification_report(y, np.argmax(oof_blend * w, axis=1), target_names=CLASSES, digits=4))

    te = pd.read_csv(os.path.join(ROOT, "data", "test.csv"))
    labels = np.argmax(test_blend * w, axis=1)
    sub = pd.DataFrame({"id": te["id"], "health_condition": [I2C[i] for i in labels]})
    out = os.path.join(SUB, "sub_ensemble.csv")
    sub.to_csv(out, index=False)
    print("\nwrote", out, "\n", sub["health_condition"].value_counts(normalize=True).round(4).to_dict())


if __name__ == "__main__":
    main()
