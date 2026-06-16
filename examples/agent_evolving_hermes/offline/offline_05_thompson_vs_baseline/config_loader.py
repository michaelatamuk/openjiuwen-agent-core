# coding: utf-8
"""Load DemoConfig from config.json."""
from __future__ import annotations

import json
import os
from pathlib import Path

from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.demo_config import (
    ALL_MODES,
    DemoConfig,
)

# Default location: config.json sits next to runner.py, one level above demo/
_DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.json"


def load_config(config_path: Path | None = None) -> DemoConfig:
    """Read *config_path* (default: ``config.json`` next to ``runner.py``) and
    return a fully populated :class:`DemoConfig`.

    The environment variable ``DEEPSEEK_API_KEY`` overrides the ``api_key``
    field in the JSON file when set.
    """
    path = Path(config_path) if config_path is not None else _DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            "Create it or pass an explicit path to load_config()."
        )

    data: dict = json.loads(path.read_text(encoding="utf-8"))

    config = DemoConfig(
        scenario_names=data["scenarios"],
        run_modes=data.get("run_modes", list(ALL_MODES)),
        api_key=os.environ.get("DEEPSEEK_API_KEY", data.get("api_key", "")),
        model=data["model"],
        api_base=data.get("api_base", "https://api.deepseek.com"),
        iterations=int(data.get("iterations", 5)),
        ts_batch_size=int(data.get("ts_batch_size", 4)),
        n_runs=int(data.get("n_runs", 1)),
        verbose=bool(data.get("verbose", False)),
        print_skill_diff=bool(data.get("print_skill_diff", False)),
        fitness_metrics=list(data.get("fitness_metrics", ["bag_of_words"])),
        oracle_data_dir=data.get("oracle_data_dir"),
    )

    print(f"{'═' * 70}")
    print(f"Model             : {config.model}  ({config.api_base})")
    print(f"GEPA iterations   : {config.iterations}")
    print(f"TS batch size     : {config.ts_batch_size}")
    print(f"Runs per mode     : {config.n_runs}")
    print(f"Fitness metrics   : {config.fitness_metrics}")
    print(f"Verbose           : {config.verbose}")
    print(f"Print skill diff  : {config.print_skill_diff}")
    print(f"Oracle data dir   : {config.oracle_data_dir or '(disabled)'}")
    print()

    os.environ.setdefault("DEEPSEEK_API_KEY", config.api_key)
    os.environ.setdefault("DEEPSEEK_API_BASE", config.api_base)

    return config
