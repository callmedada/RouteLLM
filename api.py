from __future__ import annotations

from typing import List, Dict, Optional, Tuple

from fastapi import Depends, FastAPI, BackgroundTasks, status  # type: ignore
from fastapi.routing import APIRouter  # type: ignore
from pydantic import BaseModel, Field  # type: ignore

from routellm.pipeline import RouterPipeline, TrainConfig
from routellm.train_eval import train_and_eval, train_and_eval_from_routerbench
from routellm.data import TrainItem as DataTrainItem, parse_routerbench
from routellm.logger import RunLogger


class TrainItem(BaseModel):
    text: str
    features: Dict[str, str] = Field(default_factory=dict)


class TrainRequest(BaseModel):
    items: List[TrainItem]
    model_names: List[str]
    model_costs: List[float]
    num_clusters: int = 4
    beta: float = 0.1
    lambda_soft: float = 0.5
    objective: str = "utility"  # 或 "min_cost"
    quality_threshold: float = 0.0
    fuser_type: str = "mlp"  # mlp | attention


class PredictRequest(BaseModel):
    text: str
    features: Dict[str, str] = Field(default_factory=dict)


class PredictResponse(BaseModel):
    model_name: str
    probability: float
    model_index: int


def get_pipeline() -> RouterPipeline:
    global _PIPELINE
    if _PIPELINE is None:
        # 懒初始化最小占位，训练接口会重建
        _PIPELINE = RouterPipeline(["model-a"], [0.01], TrainConfig(num_clusters=2, beta=0.1, lambda_soft=0.5))
    return _PIPELINE


router = APIRouter(prefix="/router", tags=["router"])


@router.post("/train", status_code=status.HTTP_201_CREATED)
def train(req: TrainRequest, background: BackgroundTasks) -> Dict[str, str]:
    def _do_train() -> None:
        global _PIPELINE
        config = TrainConfig(
            num_clusters=req.num_clusters,
            beta=req.beta,
            lambda_soft=req.lambda_soft,
            objective=req.objective,
            quality_threshold=req.quality_threshold,
            fuser_type=req.fuser_type,
        )
        pipeline = RouterPipeline(req.model_names, req.model_costs, config)
        texts = [it.text for it in req.items]
        feats = [it.features for it in req.items]
        pipeline.fit(texts, feats)
        _PIPELINE = pipeline

    background.add_task(_do_train)
    return {"message": "training started"}


@router.post("/evaluate")
def evaluate(req: TrainRequest) -> Dict[str, float]:
    items = [DataTrainItem(text=it.text, features=it.features) for it in req.items]
    config = TrainConfig(
        num_clusters=req.num_clusters,
        beta=req.beta,
        lambda_soft=req.lambda_soft,
        objective=req.objective,
        quality_threshold=req.quality_threshold,
        fuser_type=req.fuser_type,
    )
    metrics = train_and_eval(items, req.model_names, req.model_costs, config)
    return metrics


@router.post("/train_default", status_code=status.HTTP_201_CREATED)
def train_default(background: BackgroundTasks) -> Dict[str, str]:
    # 假设默认数据位于项目根目录 routerbench_0shot.pkl
    default_path = "routerbench_0shot.pkl"
    def _do_train_default() -> None:
        global _PIPELINE
        # 默认配置
        config = TrainConfig(num_clusters=4, beta=0.1, lambda_soft=0.5, objective="min_cost", quality_threshold=0.6)
        # 从数据集中解析模型名与均值成本
        from routellm.data import parse_routerbench, extract_xy
        items, model_names, model_costs = parse_routerbench(default_path)
        # 直接训练+评估一次以生成模型
        metrics = train_and_eval_from_routerbench(default_path, model_names, model_costs, config)
        # 用全量训练
        texts = [it.text for it in items]
        feats = [it.features for it in items]
        pipeline = RouterPipeline(model_names, model_costs, config)
        pipeline.fit(texts, feats)
        _PIPELINE = pipeline
    background.add_task(_do_train_default)
    return {"message": "default training started"}


@router.get("/evaluate_default")
def evaluate_default() -> Dict[str, float]:
    default_path = "routerbench_0shot.pkl"
    config = TrainConfig(num_clusters=4, beta=0.1, lambda_soft=0.5, objective="min_cost", quality_threshold=0.6)
    return train_and_eval_from_routerbench(default_path, None, None, config)


@router.get("/runs/{run_id}/metrics")
def get_metrics(run_id: str) -> Dict[str, float]:
    import json
    from pathlib import Path

    path = Path("runs") / run_id / "metrics.json"
    if not path.exists():
        return {"error": "not_found"}
    return json.loads(path.read_text(encoding="utf-8"))


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, pipeline: RouterPipeline = Depends(get_pipeline)) -> PredictResponse:
    name, prob, idx = pipeline.predict(req.text, req.features)
    return PredictResponse(model_name=name, probability=prob, model_index=idx)


_PIPELINE: Optional[RouterPipeline] = None

app = FastAPI(title="RouteLLM Router API")
app.include_router(router)

