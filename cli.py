import argparse
from pathlib import Path
from routellm.search import grid_search_routerbench
from routellm.logger import RunLogger


def main() -> None:
    parser = argparse.ArgumentParser(description="RouteLLM grid search on routerbench_0shot.pkl")
    parser.add_argument("--data", default="routerbench_0shot.pkl")
    parser.add_argument("--lambdas", default="0.2,0.5,0.8")
    parser.add_argument("--thresholds", default="0.4,0.6,0.8")
    parser.add_argument("--clusters", default="4")
    args = parser.parse_args()

    lambda_list = [float(x) for x in args.lambdas.split(",") if x]
    threshold_list = [float(x) for x in args.thresholds.split(",") if x]
    clusters = [int(x) for x in args.clusters.split(",") if x]

    results = grid_search_routerbench(args.data, lambda_list, threshold_list, num_clusters_list=clusters)

    logger = RunLogger()
    run = logger.create(prefix="grid")
    best_metrics, best_cfg = results[0]

    # 保存最佳
    (run.run_dir / "best_config.txt").write_text(str(best_cfg), encoding="utf-8")
    logger.log_metrics(run, best_metrics)

    # 保存全部结果
    import json

    (run.run_dir / "all_results.json").write_text(
        json.dumps([
            {"metrics": m, "config": c.__dict__} for m, c in results
        ], ensure_ascii=False),
        encoding="utf-8",
    )
    print("Best:", best_metrics, best_cfg)


if __name__ == "__main__":
    main()