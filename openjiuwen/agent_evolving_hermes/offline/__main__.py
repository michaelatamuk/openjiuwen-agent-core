# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Entry point for `python -m openjiuwen.agent_evolving_hermes`.

Invokes the offline GEPA skill evolver CLI.

Examples:
    python -m openjiuwen.agent_evolving_hermes.offline --skill git-review
    python -m openjiuwen.agent_evolving_hermes.offline --skill git-review --dry-run
    python -m openjiuwen.agent_evolving_hermes.offline --skill git-review --iterations 5 --reuse-dataset
    python -m openjiuwen.agent_evolving_hermes.offline --help
"""
from openjiuwen.agent_evolving_hermes.offline.cli import main

if __name__ == "__main__":
    main()
