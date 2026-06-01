
from __future__ import annotations

from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.writer_skill import \
    _write_skill


def step(skills_root, skill_name, skill_frontmatter, skill_body) -> None:
    _write_skill(skills_root, skill_name, skill_frontmatter, skill_body)
