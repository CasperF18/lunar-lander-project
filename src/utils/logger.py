from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RunConfig:
    run_name: str
    env_id: str
    seed: int
    max_steps_per_episode: int
    notes: str = ""
    extra: Optional[Dict[str, Any]] = None      #This hopefully becomes our parameters


class RunLogger:
    """"
    Writes:
        - run metadata:         logs/<run_name>/config.json
        - per-episode logs:     logs/<run_name>/episodes.jsonl
        - summary csv:          logs/<run_name>/summary.csv
    """
    def __init__(self, config: RunConfig):
        root_dir = "logs"
        ts = time.strftime("%Y%m%d-%H%M%S")
        safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in config.run_name)
        self.run_dir = PROJECT_ROOT / root_dir / f"{ts}_{safe_name}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.config = config
        self.episodes_path = self.run_dir / "episodes.jsonl"
        self.summary_path = self.run_dir / "summary.csv"
        self.config_path = self.run_dir / "config.json"

        # Write config once
        with self.config_path.open("w", encoding="utf-8") as f:
            json.dump(asdict(config), f, indent=2)

        # Prepare summary CSV
        self._csv_file = self.summary_path.open("w", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=[
            "episode", "return", "length", "terminated", "truncated"
        ])
        self._csv_writer.writeheader()

        self._episode_count = 0

    def log_episode(self, episode_return: float, episode_length: int, terminated: bool, truncated: bool, info: Optional[Dict[str, Any]] = None):
        record = {
            "episode": self._episode_count,
            "return": float(episode_return),
            "length": int(episode_length),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "info": info or {},
        }

        # JSONL (one JSON object per line)
        with self.episodes_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        # CSV summary
        self._csv_writer.writerow({
            "episode": record["episode"],
            "return": record["return"],
            "length": record["length"],
            "terminated": record["terminated"],
            "truncated": record["truncated"],
        })
        self._csv_file.flush()

        self._episode_count += 1

    def close(self):
        try:
            self._csv_file.close()
        except Exception:
            pass
