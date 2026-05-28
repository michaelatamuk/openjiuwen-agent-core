# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Session log mining for eval dataset construction.

Mirrors hermes-agent-self-evolution evolution/core/external_importers.py.
Key Jiuwen adaptation: JiuwenSessionImporter reads ~/.jiuwen/sessions/*.json
(same format as HermesSessionImporter but different path).
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List

from .jiuwen_session_importer import JiuwenSessionImporter
from .claude_code_importer import ClaudeCodeImporter
from .skill_example_extractor import SkillExampleExtractor

if TYPE_CHECKING:
    from openjiuwen.agent_evolving_hermess.offline.dataset_builder import EvalDataset, EvalExample


def build_dataset_from_external(
    skill_name: str,
    skill_text: str,
    sources: List[str],
    output_path: Path,
    model: str,
    max_examples: int = 50,
) -> "EvalDataset":
    from openjiuwen.agent_evolving_hermess.offline.dataset_builder import EvalDataset

    all_messages: List[Dict] = []
    importers = {
        "jiuwen": ("Jiuwen sessions", JiuwenSessionImporter),
        "claude-code": ("Claude Code history", ClaudeCodeImporter),
    }
    for source in sources:
        if source in importers:
            _label, cls = importers[source]
            msgs = cls.extract_messages()
            all_messages.extend(msgs)
    if not all_messages:
        return EvalDataset()
    extractor = SkillExampleExtractor(model=model)
    examples = extractor.extract_examples(all_messages, skill_name, skill_text, max_examples)
    if not examples:
        return EvalDataset()
    random.shuffle(examples)
    n = len(examples)
    n_train = max(1, int(n * 0.5))
    n_val = max(1, int(n * 0.25))
    ds = EvalDataset(
        train=examples[:n_train],
        val=examples[n_train: n_train + n_val],
        holdout=examples[n_train + n_val:],
    )
    ds.save(output_path)
    return ds
