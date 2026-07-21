# TabPFN GPU kernel — orthogonal signal for S6E7

TabPFN is a tabular **foundation model**; its inductive bias is orthogonal to the GBDT ecosystem,
and (per [discussion #726672](https://www.kaggle.com/competitions/playground-series-s6e7/discussion/726672))
it is the one model family shown to carry *incremental* signal past the ~0.950 GBDT Bayes frontier
on this dataset. It needs a GPU, so it runs as a Kaggle kernel rather than locally.

- **Kernel:** https://www.kaggle.com/code/shreyascppsc/s6e7-tabpfn-probs (GPU + internet on)
- **Method:** TabPFN's native context limit is ~10k rows, so we fit a *subsample ensemble*
  (`N_ENS` models on class-balanced ~10k subsamples, averaged) for 5-fold OOF + test predictions.
- **Outputs:** `oof_tabpfn.npy`, `test_tabpfn.npy`, `sub_tabpfn.csv`, `meta.json`.

## Pull the outputs when the run finishes, then blend locally
```bash
cd ..                                   # project root
kaggle kernels output shreyascppsc/s6e7-tabpfn-probs -p artifacts
python src/stack.py lgbm xgb histgb B_rule tabpfn   # logistic meta-model over all OOF sets
# submit submissions/sub_stack.csv if OOF balanced accuracy improves over the GBM-only stack
```
Rationale: adding a *correlated* GBM buys nothing (already verified); a single orthogonal family
should. Keep the honest CV as the yardstick — the private board is what counts.
