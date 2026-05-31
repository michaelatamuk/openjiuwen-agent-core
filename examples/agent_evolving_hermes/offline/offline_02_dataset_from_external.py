# coding: utf-8
"""Example 4: Build eval dataset from external session logs.

Mirrors Hermes's two-stage relevance filter:
  Stage 1 (cheap): keyword heuristic — does the message mention the skill name
                   or keywords from the skill text?
  Stage 2 (LLM):  ScoreRelevance DSPy signature — is the message truly relevant?
                   Returns: {relevant, expected_behavior, difficulty, category}

Sources supported:
  - "jiuwen"      → ~/.jiuwen/sessions/*.json
  - "claude-code" → ~/.claude/history.jsonl

Usage:
    python examples/agent_evolving_hermess/04_dataset_from_external.py
"""
from pathlib import Path

from openjiuwen.agent_evolving_hermes.offline import (
    ClaudeCodeImporter,
    JiuwenSessionImporter,
    build_dataset_from_external,
)


def example_mine_sessions():
    """Show how many messages each importer finds locally."""
    print("── Scanning local session logs ──────────────────────")

    jiuwen_msgs = JiuwenSessionImporter.extract_messages(limit=20)
    print(f"  Jiuwen sessions   : {len(jiuwen_msgs)} user messages found")
    if jiuwen_msgs:
        sample = jiuwen_msgs[0]
        print(f"    Sample input    : {sample['task_input'][:80]}...")
        print(f"    Source          : {sample['source']}")

    cc_msgs = ClaudeCodeImporter.extract_messages(limit=20)
    print(f"  Claude Code history: {len(cc_msgs)} entries found")
    if cc_msgs:
        sample = cc_msgs[0]
        print(f"    Sample input    : {sample['task_input'][:80]}...")


def example_build_dataset():
    """Build an eval dataset for a skill from external logs.

    Requires:
      - A skill at ~/.jiuwen/skills/git-review/SKILL.md
      - An LLM API key (OPENAI_API_KEY etc.)
      - Session logs at ~/.jiuwen/sessions/ or ~/.claude/history.jsonl
    """
    skill_name = "git-review"
    skills_root = Path.home() / ".jiuwen" / "skills"
    output_path = Path("./skill_evolver_output") / skill_name / "external_dataset"

    # Read skill text
    skill_path = skills_root / skill_name / "SKILL.md"
    if not skill_path.exists():
        print(f"Skill not found at {skill_path} — skipping dataset build example.")
        return

    skill_text = skill_path.read_text(encoding="utf-8")
    print(f"\nBuilding external dataset for skill '{skill_name}'...")
    print(f"  Skill chars : {len(skill_text)}")

    dataset = build_dataset_from_external(
        skill_name=skill_name,
        skill_text=skill_text,
        sources=["jiuwen", "claude-code"],
        output_path=output_path,
        model="openai/gpt-4.1-mini",   # relevance scoring model
        max_examples=20,
    )

    print(f"\n── Dataset built ─────────────────────────────────────")
    print(f"  Train   : {len(dataset.train)} examples")
    print(f"  Val     : {len(dataset.val)} examples")
    print(f"  Holdout : {len(dataset.holdout)} examples")

    if dataset.train:
        print(f"\n  Sample training example:")
        ex = dataset.train[0]
        print(f"    input            : {ex.task_input[:80]}...")
        print(f"    expected_behavior: {ex.expected_behavior[:80]}...")
        print(f"    difficulty        : {ex.difficulty}")
        print(f"    source            : {ex.source}")

        print(f"\n  Dataset saved to: {output_path}")


def example_relevance_filter_standalone():
    """Demonstrate the two-stage relevance filter on synthetic data."""
    print("\n── Relevance filter (heuristic stage only) ──────────")

    # Simulate messages from session logs
    messages = [
        {"source": "jiuwen", "task_input": "Review this git diff and tell me what's wrong.", "assistant_response": ""},
        {"source": "jiuwen", "task_input": "What is the capital of France?", "assistant_response": "Paris."},
        {"source": "claude-code", "task_input": "Check my PR for potential security issues in the auth module.", "assistant_response": ""},
        {"source": "jiuwen", "task_input": "How do I make pasta?", "assistant_response": "Boil water..."},
        {"source": "claude-code", "task_input": "git diff HEAD~1 looks huge, can you review the changes?", "assistant_response": ""},
    ]

    skill_name = "git-review"
    skill_text = "Guidelines for reviewing git diffs and pull requests."

    # Stage 1: heuristic filter (no LLM cost)
    from openjiuwen.agent_evolving_hermes.offline.external_importers.relevance_filter import _is_relevant_to_skill
    for msg in messages:
        relevant = _is_relevant_to_skill(msg["task_input"], skill_name, skill_text)
        mark = "✓" if relevant else "✗"
        print(f"  {mark} {msg['task_input'][:60]}")

    print(
        "\n  (Stage 2 LLM scoring would further filter these "
        "using ScoreRelevance DSPy signature — requires API key)"
    )


if __name__ == "__main__":
    example_mine_sessions()
    example_relevance_filter_standalone()
    example_build_dataset()
