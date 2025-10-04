# RouteLLM (OOP Pipeline)

A modular LLM router that sends each query to the cheapest model that meets a quality threshold. The pipeline fuses two branches: LIMBO (feature-based) and BERT (dense embedding), then learns a small PyTorch fuser (MLP or Attention) with cost‑aware soft labels.

## Install
```bash
pip install -U numpy torch fastapi uvicorn pandas sentence-transformers  # optional
pip install limbo-cluster  # LIMBO package

```

```bash

#  uv 安装
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv .venv
source .venv/bin/activate
uv pip install -e '.[dev,sentence,uvicorn,limbo]'
uv run pytest -q
```


## Dataset
- Default dataset: `routerbench_0shot.pkl` (pandas DataFrame). Loader parses model columns and per‑model total costs automatically.

## Quick Start
```python
from routellm import RouterPipeline, TrainConfig
from routellm.data import parse_routerbench

items, model_names, model_costs = parse_routerbench("routerbench_0shot.pkl")
texts = [it.text for it in items]
feats  = [it.features for it in items]

cfg = TrainConfig(num_clusters=4, beta=0.1, lambda_soft=0.5, objective="min_cost", quality_threshold=0.6, fuser_type="mlp")
pipeline = RouterPipeline(model_names, model_costs, cfg)
pipeline.fit(texts, feats)
print(pipeline.predict("Explain attention", {"eval": "demo"}))
```

## CLI (Grid Search)
```bash
python cli.py --data routerbench_0shot.pkl --lambdas 0.3,0.5 --thresholds 0.5,0.7 --clusters 3
```
- Results saved into `runs/<run_id>/`.

## REST API
```bash
uvicorn api:app --reload --port 8000
```
- POST `/router/train_default` – start background training
- GET `/router/evaluate_default` – quick eval
- POST `/router/predict` – single query routing

## Structure
```
routellm/
  core/            # config, logger, mapping
  branches/        # limbo & bert branches
  models/          # encoders, torch fusers
  data.py          # loaders, split helpers
  pipeline.py      # end-to-end router
  train_eval.py    # train/eval helpers
  search.py        # grid search
api.py              # FastAPI service
cli.py              # Grid-search runner
```

## Notes
- All clustering uses `limbo_cluster.LimboAgglomerative`.
- If `sentence-transformers` is missing, BERT branch falls back to a hash encoder.