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

## Layout
```
src/train.py        # end-to-end: load -> CV -> tune decision rule -> submission
src/eda.py          # data exploration
data/               # competition CSVs (gitignored)
submissions/        # generated submissions (gitignored)
```

## Run
```bash
python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
kaggle competitions download -c playground-series-s6e7 -p data && (cd data && unzip -o '*.zip')
python src/train.py
kaggle competitions submit -c playground-series-s6e7 -f submissions/sub.csv -m "..."
```
