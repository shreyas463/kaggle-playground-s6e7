#!/usr/bin/env bash
# Pull TabPFN kernel outputs once the run completes, then blend + evaluate.
set -e
cd "$(dirname "$0")"
. .venv/bin/activate
echo "== kernel status =="
kaggle kernels status shreyascppsc/s6e7-tabpfn-probs
echo "== pulling outputs -> artifacts/ =="
kaggle kernels output shreyascppsc/s6e7-tabpfn-probs -p artifacts
echo "== meta =="; cat artifacts/meta.json 2>/dev/null || true
echo "== blend: GBMs + rule + TabPFN =="
python src/stack.py lgbm xgb histgb B_rule tabpfn
