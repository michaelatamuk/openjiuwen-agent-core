# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Constraint validation for evolved skill candidates.

Mirrors hermes-agent-self-evolution evolution/core/constraints.py exactly.
"""
from __future__ import annotations

import subprocess
from typing import List, Optional

import yaml

from skill_evolver_stages.stage02_skill_constraint_validator.constraint_result import ConstraintResult

from openjiuwen.agent_evolving_hermes.offline.evolvers.skill_evolver_config import EvolverConfig


class ConstraintValidator:
    """Validates evolved skill text against all hard constraints.

    Every constraint must pass before an evolved skill is accepted.
    """

    def __init__(self, config: "EvolverConfig"):
        self.config = config

    def validate_all(self, evolved_text: str, artifact_type: str = "skill", baseline_text: Optional[str] = None) \
            -> List[ConstraintResult]:
        """Run all constraints. Returns list of ConstraintResult (all must be passed=True)."""
        results = []

        # 1. Non-empty
        results.append(self._check_non_empty(evolved_text))

        # 2. Size limit
        results.append(self._check_size(evolved_text))

        # 3. Growth limit (only if baseline provided)
        if baseline_text is not None:
            results.append(self._check_growth(evolved_text, baseline_text))

        # 4. Skill structure (frontmatter) — only for skills
        if artifact_type == "skill":
            results.append(self._check_skill_structure(evolved_text))

        # 5. Test suite (opt-in)
        if self.config.run_pytest:
            results.append(self._check_test_suite())

        return results

    @staticmethod
    def _check_non_empty(text: str) -> ConstraintResult:
        if text and text.strip():
            return ConstraintResult(True, "non_empty", "Content is non-empty.")
        return ConstraintResult(False, "non_empty", "Evolved artifact is empty.")

    def _check_size(self, text: str) -> ConstraintResult:
        limit = self.config.max_skill_size
        if len(text) <= limit:
            return ConstraintResult(True, "size_limit", f"Size {len(text)} ≤ {limit}.")

        return ConstraintResult(False, "size_limit",
                                f"Size {len(text)} exceeds limit {limit} by {len(text) - limit} chars.")

    def _check_growth(self, evolved: str, baseline: str) -> ConstraintResult:
        if not baseline:
            return ConstraintResult(True, "growth_limit", "No baseline — skipping growth check.")

        growth = (len(evolved) - len(baseline)) / max(1, len(baseline))
        limit = self.config.max_prompt_growth
        if growth <= limit:
            return ConstraintResult(True, "growth_limit", f"Growth {growth:.1%} ≤ {limit:.0%}.")

        return ConstraintResult(False, "growth_limit", f"Growth {growth:.1%} exceeds limit {limit:.0%}.")

    @staticmethod
    def _check_skill_structure(text: str) -> ConstraintResult:
        if not text.startswith("---"):
            return ConstraintResult(False, "skill_structure", "Missing YAML frontmatter (---).")

        end = text.find("\n---", 3)
        if end == -1:
            return ConstraintResult(False, "skill_structure", "Frontmatter not closed (missing closing ---).")

        try:
            fm = yaml.safe_load(text[3:end]) or {}
        except yaml.YAMLError as e:
            return ConstraintResult(False, "skill_structure", f"Invalid YAML frontmatter: {e}")

        if not fm.get("name"):
            return ConstraintResult(False, "skill_structure", "Frontmatter missing 'name' field.")

        if not fm.get("description"):
            return ConstraintResult(False, "skill_structure", "Frontmatter missing 'description' field.")

        return ConstraintResult(True, "skill_structure", "Frontmatter is valid.")

    def _check_test_suite(self) -> ConstraintResult:
        try:
            result = subprocess.run(["pytest", "tests/", "-q", "--tb=no"], capture_output=True, text=True,
                                    timeout=self.config.pytest_timeout,)
            if result.returncode == 0:
                return ConstraintResult(True, "test_suite", "All tests passed.")

            return ConstraintResult(False, "test_suite",
                f"Tests failed (exit {result.returncode}):\n{result.stdout[-500:]}",)
        except subprocess.TimeoutExpired:
            return ConstraintResult(False, "test_suite", f"Tests timed out after {self.config.pytest_timeout}s.")
        except FileNotFoundError:
            return ConstraintResult(True, "test_suite", "pytest not found — skipping.")
