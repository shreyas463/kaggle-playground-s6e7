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

**Best submission so far:** single balanced LightGBM — public LB **0.94960**.

### Next steps to close the gap
- Recover / identify the real source dataset behind the synthetic data (often the biggest lever in
  Playground competitions) and use it as extra training signal.
- Targeted feature discovery around the exact decision boundaries (sleep/stress/activity thresholds).
- Per-fold threshold optimization and calibrated stacking with a diverse base (e.g. RandomForest,
  which was competitive here) rather than correlated GBMs.

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
