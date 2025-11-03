import argparse
from pathlib import Path
from routellm.search import grid_search_routerbench
from routellm.logger import RunLogger
from routellm.pipeline import RouterPipeline
from routellm.data import parse_routerbench


def main() -> None:
    parser = argparse.ArgumentParser(description="RouteLLM grid search on routerbench_0shot.pkl")
    parser.add_argument("--data", default="routerbench_0shot.pkl")
    parser.add_argument("--lambdas", default="0.2,0.5,0.8")
    parser.add_argument("--thresholds", default="0.4,0.6,0.8")
    parser.add_argument("--clusters", default="4")
    parser.add_argument("--fuser_types", default="mlp")
    parser.add_argument("--fusion_betas", default="")
    parser.add_argument("--limbo_taus", default="")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--train_full", action="store_true", help="使用最佳配置对全量数据训练")
    parser.add_argument("--save_model", default="", help="保存管道模型到该路径（JSON）")
    parser.add_argument(
        "--limbo_input_dump",
        default="",
        help="将 LIMBO 输入数据导出到指定路径（JSON 文件或目录）",
    )
    parser.add_argument("--spacy_model", default="en_core_web_sm", help="必选：spaCy 模型名称（需提前下载）")
    parser.add_argument(
        "--spacy_components",
        default="tok2vec,tagger,parser,ner",
        help="启用的 spaCy pipeline 组件，逗号分隔",
    )
    parser.add_argument("--query_hash_buckets", type=int, default=128, help="查询哈希特征的桶数")
    parser.add_argument(
        "--query_max_hash_per_ngram",
        type=int,
        default=64,
        help="每种 n-gram 保留的哈希桶上限",
    )
    parser.add_argument(
        "--query_embed_k",
        type=int,
        default=0,
        help="基于 BERT 表示的嵌入簇数量（<=0 表示禁用）",
    )
    args = parser.parse_args()

    lambda_list = [float(x) for x in args.lambdas.split(",") if x]
    threshold_list = [float(x) for x in args.thresholds.split(",") if x]
    clusters = [int(x) for x in args.clusters.split(",") if x]
    fuser_types = [x for x in args.fuser_types.split(",") if x]
    fusion_betas = [float(x) for x in args.fusion_betas.split(",") if x]
    limbo_taus = [float(x) if x.lower() != "none" else None for x in args.limbo_taus.split(",") if x]
    spacy_components = tuple(comp.strip() for comp in args.spacy_components.split(",") if comp.strip())
    embed_k = args.query_embed_k if args.query_embed_k > 0 else None

    results = grid_search_routerbench(
        args.data,
        lambda_list,
        threshold_list,
        fuser_types=fuser_types,
        num_clusters_list=clusters,
        fusion_betas=(fusion_betas if fusion_betas else None),
        limbo_taus=(limbo_taus if limbo_taus else None),
        val_ratio=args.val_ratio,
        use_query_features=True,
        spacy_model=args.spacy_model,
        spacy_components=spacy_components if spacy_components else None,
        query_hash_buckets=args.query_hash_buckets,
        query_max_hash_per_ngram=args.query_max_hash_per_ngram,
        query_embed_k=embed_k,
    )

    print("[cli] grid search finished, selecting best ...", flush=True)
    logger = RunLogger()
    run = logger.create(prefix="grid")
    best_metrics, best_cfg = results[0]

    limbo_dump_path = args.limbo_input_dump.strip() or None
    if limbo_dump_path is not None:
        best_cfg.limbo_input_dump_path = limbo_dump_path

    best_cfg.use_query_features = True
    best_cfg.use_spacy_features = True
    best_cfg.spacy_model = args.spacy_model
    if spacy_components:
        best_cfg.spacy_enable_components = spacy_components
    best_cfg.query_hash_buckets = args.query_hash_buckets
    best_cfg.query_max_hash_per_ngram = args.query_max_hash_per_ngram
    best_cfg.query_embed_k = embed_k

    # 保存最佳
    (run.run_dir / "best_config.txt").write_text(str(best_cfg), encoding="utf-8")
    # 将最佳配置写入 metrics.json
    logger.log_metrics(run, best_metrics, config=best_cfg)

    # 保存全部结果
    import json

    (run.run_dir / "all_results.json").write_text(
        json.dumps([
            {"metrics": m, "config": c.__dict__} for m, c in results
        ], ensure_ascii=False),
        encoding="utf-8",
    )
    print("Best:", best_metrics, best_cfg)

    # 选择性：基于最佳配置进行全量训练并可保存模型
    if args.train_full or args.save_model:
        items, model_names, model_costs = parse_routerbench(args.data)
        texts = [it.text for it in items]
        feats = [it.features for it in items]
        pipe = RouterPipeline(model_names, model_costs, best_cfg)
        pipe.fit(texts, feats)
        if args.save_model:
            Path(args.save_model).parent.mkdir(parents=True, exist_ok=True)
            pipe.save(args.save_model)
            print(f"Model saved to {args.save_model}")
    elif limbo_dump_path is not None:
        print("[cli] 已设置 limbo_input_dump，但未进行全量训练，未生成导出文件", flush=True)


if __name__ == "__main__":
    main()