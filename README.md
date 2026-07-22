# Kaggle Playground S6E7 — Predicting Student Health Risk

Solution for [Playground Series S6E7](https://www.kaggle.com/competitions/playground-series-s6e7)
(ends 2026-07-31). Predict `health_condition` ∈ {`at-risk`, `unhealthy`, `fit`} for students.

## Problem
- **Task:** 3-class classification. Train 690k rows, test 296k, 13 features (7 numeric, 6 categorical),
  with missing values throughout.
- **Metric:** **Balanced accuracy** (mean per-class recall) — the classes are very imbalanced
  (at-risk 86%, unhealthy 8%, fit 6%), so the rare classes matter as much as the majority. Plain
  accuracy-optimization is the wrong objective.
- Data is synthetically generated from a real-world dataset; a few features (esp. `stress_level`,
  `physical_activity_level`) carry strong signal.

## Approach
- **Model:** LightGBM multiclass with native categorical + missing-value handling.
- **Imbalance:** class-balanced sample weights so each class contributes equally to the loss.
- **Validation:** stratified K-fold; out-of-fold (OOF) predictions scored with balanced accuracy.
- **Decision rule:** because the metric is balanced accuracy, argmax of raw probabilities is not
  optimal — per-class probability multipliers are tuned on OOF to maximize balanced accuracy, then
  applied to the averaged test-fold probabilities.

## Algorithm comparison
Balanced accuracy, 5-fold, on a 200k stratified subsample (relative ranking):

| Algorithm | Balanced accuracy |
|---|---|
| HistGradientBoosting | **0.9482** |
| RandomForest | 0.9480 |
| XGBoost | 0.9465 |
| LightGBM | 0.9452 |
| ExtraTrees | 0.9377 |
| LogisticRegression | 0.9188 |
| KNN | 0.8993 |

Gradient-boosted trees win; RandomForest is a strong close second. Linear and distance-based
models lag badly because the target is driven by **interactions** (fit = low stress *and* active
*and* good sleep; unhealthy = high stress *and* short sleep) that they can't represent.

On the **full 690k** data the boosted models all land at **~0.9495–0.9496 OOF**, which matches the
public LB almost exactly (OOF 0.9496 → LB 0.9496 — tight CV↔LB). A greedy blend of LGBM+XGB+HistGB
nudged OOF to 0.94987 but did not improve the LB (the models are too correlated), and missingness
indicators added nothing (+0.00002). The remaining gap to the leaderboard top (~0.953) is signal not
yet captured — the likely levers are **feature discovery or the original source dataset**, not more
algorithms or ensembling.

## The original dataset, found and cracked

The synthetic source is [`ziya07/college-student-health-behavior-dataset`](https://www.kaggle.com/datasets/ziya07/college-student-health-behavior-dataset)
(identified in [discussion #717222](https://www.kaggle.com/competitions/playground-series-s6e7/discussion/717222)).
Its target is **fully deterministic** — a depth-4 decision tree over 3 features (verified: accuracy
1.0 on the original 50k):

```
sleep < 6h:  stress == high → unhealthy;  else → at-risk
sleep ≥ 6h:  stress == low ∧ activity == active ∧ sleep ≥ 7h → fit;  else → at-risk
```

The competition data is this rule + feature noise + injected missingness — which explains the
~0.950 plateau: past the Bayes frontier there is nothing left to learn.

**Experiments (full data, 5-fold, tuned decision rule; reference lean LGBM = 0.94966 OOF):**

| Config | OOF | Δ | LB |
|---|---|---|---|
| + rule-threshold features (sleep<6/<7, stress, activity flags, rule label) | **0.94984** | **+0.00018** | **0.94965** ← best |
| + original 50k as extra train rows | 0.94937 | −0.00029 | — |
| + both | 0.94955 | −0.00011 | — |
| stack (LGBM+XGB+HistGB+rule) via logistic meta-model | 0.94986 | +0.00002 vs rule | — |

Lessons: giving the model the *exact* generation thresholds helps; training on the noise-free
original **hurts** (it pulls the model toward the deterministic rule, which is wrong for noisy rows
near thresholds — the original's value was revealing the rule, not its rows); stacking correlated
GBMs adds ~nothing.

## TabPFN — the orthogonal family ([kernel](https://www.kaggle.com/code/shreyascppsc/s6e7-tabpfn-probs))

TabPFN is a tabular **foundation model** whose errors are orthogonal to the GBDT ecosystem — the one
family shown to carry incremental signal past the GBDT frontier. Ran it on a Kaggle P100 GPU
(subsample ensemble: 6 TabPFN-v2 models on class-balanced 8k subsamples, 5-fold OOF + test).

| Model / blend | OOF | LB |
|---|---|---|
| TabPFN v2 alone | 0.9472 | — |
| stack: GBMs + rule features | 0.94986 | — |
| **stack: GBMs + rule + TabPFN** | **0.94990** | **0.94983 ← best** |

TabPFN alone is individually *weaker* (0.9472), but adding it to the logistic stack nudged OOF
(+0.00004) and, more tellingly, lifted the **public LB to 0.94983** (best of all submissions). Small,
but the direction confirms the orthogonality thesis: correlated GBMs are exhausted; a genuinely
different inductive bias is the only thing that still moves an honest blend.

Getting TabPFN running on the kernel meant clearing three real blockers, documented in
[`kernel/`](kernel/): competition data mounts under `/kaggle/input/competitions/…`; the pip default
is the *gated* v3 model (force ungated v2 via `model_path`); and Kaggle's torch 2.10+cu128 dropped
Pascal (sm_60) kernels while assigning a P100 — pin torch 2.5.1+cu121 before import.

### Context on the leaderboard (from the top scorer's own writeup)
The public 0.952+ scores come from **public-LB probing** — the public split is deterministic, so
row-level membership can be recovered by group testing. That transfers zero to the private 80%.
The honest CV ceiling is **~0.9508**; our honest ~0.950 blend may rank well on private when the
probed scores collapse. The submitted stack is CV-selected (no LB feedback), i.e. the "defense"
ledger in that framework.

## Layout
```
src/features.py           # feature engineering + design matrices per model family
src/train.py              # single LightGBM end-to-end -> submission
src/train_models.py       # full-data multi-model OOF (LGBM/XGB/CatBoost/HistGB/RF/ET/LogReg/KNN)
src/compare_algos.py      # fast apples-to-apples algorithm comparison (subsample)
src/ensemble.py           # greedy blend + balanced-accuracy decision-rule tuning
src/experiment_features.py# feature ablations (e.g. missingness indicators)
data/  submissions/  artifacts/   # gitignored
```

## Run
```bash
python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
kaggle competitions download -c playground-series-s6e7 -p data && (cd data && unzip -o '*.zip')
python src/train.py
kaggle competitions submit -c playground-series-s6e7 -f submissions/sub.csv -m "..."
```
