from __future__ import annotations

from routellm.pipeline import RouterPipeline, TrainConfig


def test_pipeline_train_predict() -> None:
    texts = [
        "I love machine learning and NLP.",
        "请解释一下Transformer的自注意力机制。",
        "What is the capital of France?",
        "负样本挖掘如何提升检索质量？",
    ]
    feats = [
        {"sentiment": "pos", "factual": "high", "domain": "ml"},
        {"sentiment": "neutral", "factual": "high", "domain": "ml"},
        {"sentiment": "neutral", "factual": "high", "domain": "trivia"},
        {"sentiment": "neutral", "factual": "medium", "domain": "ir"},
    ]
    models = ["gpt-3.5-turbo", "gpt-4", "claude-3", "llama-2"]
    costs = [0.001, 0.03, 0.015, 0.002]

    cfg = TrainConfig(num_clusters=2, beta=0.1, lambda_soft=0.5, prefer_transformer=False)
    pipeline = RouterPipeline(models, costs, cfg)
    pipeline.fit(texts, feats)

    name, prob, idx = pipeline.predict("Explain attention in transformers", {"sentiment": "neutral", "factual": "high", "domain": "ml"})
    assert name in models
    assert 0 <= prob <= 1
    assert 0 <= idx < len(models)

