from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class RunInfo:
    run_id: str
    run_dir: Path


class RunLogger:
    def __init__(self, base_dir: str | Path = "runs") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create(self, run_id: Optional[str] = None, prefix: str = "router") -> RunInfo:
        if not run_id:
            ts = time.strftime("%Y%m%d-%H%M%S")
            run_id = f"{prefix}-{ts}"
        run_dir = self.base_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return RunInfo(run_id=run_id, run_dir=run_dir)

    def log_metrics(self, run: RunInfo, metrics: Dict[str, float], *, config: Optional[Any] = None, extra: Optional[Dict[str, Any]] = None) -> None:
        csv_path = run.run_dir / "metrics.csv"
        json_path = run.run_dir / "metrics.json"

        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for k, v in metrics.items():
                writer.writerow([k, v])

        with json_path.open("w", encoding="utf-8") as f:
            payload: Dict[str, Any] = {"metrics": metrics}
            if config is not None:
                if is_dataclass(config):
                    payload["config"] = asdict(config)
                elif isinstance(config, dict):
                    payload["config"] = config
                else:
                    # best-effort stringify
                    try:
                        payload["config"] = asdict(config)  # type: ignore[arg-type]
                    except Exception:
                        payload["config"] = str(config)
            if extra:
                payload.update(extra)
            json.dump(payload, f, ensure_ascii=False, indent=2)

