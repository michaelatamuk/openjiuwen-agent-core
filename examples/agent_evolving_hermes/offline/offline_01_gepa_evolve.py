# coding: utf-8
"""Example 3: Offline GEPA skill evolution.

Mirrors Hermes's GEPA (Genetic Evolution of Prompt Artifacts) offline track.

What this does:
  1. Loads a SKILL.md from ~/.jiuwen/skills/<skill_name>/SKILL.md
  2. Validates baseline constraints (size, YAML frontmatter)
  3. Generates synthetic eval cases using LLM (or loads golden/external dataset)
  4. Runs DSPy GEPA (genetic prompt search) to optimise the SKILL.md body
     Falls back to MIPROv2 if GEPA is not available
  5. Validates evolved constraints (size, growth limit, frontmatter)
  6. Scores baseline vs evolved on holdout set using LLM-as-judge
  7. Saves evolved_skill.md + metrics.json to output_dir

Requirements:
    pip install "dspy>=3.0.0" pyyaml click litellm
    export OPENAI_API_KEY=...   (or whichever model you configure)

Usage:
    python examples/agent_evolving_hermess/03_offline_gepa_evolve.py

    Or via CLI:
    python -m openjiuwen.agent_evolving_hermess.cli \\
        --skill git-review \\
        --iterations 5 \\
        --eval-source synthetic \\
        --optimizer-model openai/gpt-4o-mini \\
        --eval-model openai/gpt-4o-mini
"""
from pathlib import Path

from offline.evolvers.skill_evolver_single_params import SkillEvolverParams
from openjiuwen.agent_evolving_hermes.offline import EvolverConfig, evolve_single_skill


def run_example_evolution():
    """Run a 3-iteration GEPA evolution on a skill named 'git-review'.

    The skill must exist at:
      ~/.jiuwen/skills/git-review/SKILL.md
    OR at a category subdirectory, e.g.:
      ~/.jiuwen/skills/coding/git-review/SKILL.md

    If the skill does not exist, create a minimal one first:
      mkdir -p ~/.jiuwen/skills/git-review
      cat > ~/.jiuwen/skills/git-review/SKILL.md << 'EOF'
      ---
      name: git-review
      description: Guidelines for reviewing git diffs and pull requests.
      ---

      When reviewing a git diff or PR:
      1. Check that the changes make logical sense.
      2. Verify tests cover the new code.
      3. Look for any potential security issues.
      EOF
    """
    config = EvolverConfig(
        # ── Skill location ─────────────────────────────────────────────────
        skills_root=Path.home() / ".jiuwen" / "skills",

        # ── GEPA settings ──────────────────────────────────────────────────
        iterations=3,               # use more in production (10–50)
        population_size=5,

        # ── Models ─────────────────────────────────────────────────────────
        optimizer_model="openai/gpt-4.1",       # GEPA uses this for reflections
        eval_model="openai/gpt-4.1-mini",        # LLM-as-judge uses this
        judge_model="openai/gpt-4.1",            # dataset generation model

        # ── Constraints ────────────────────────────────────────────────────
        max_skill_size=15_000,       # 15 KB hard cap
        max_prompt_growth=0.20,      # evolved skill may not grow > 20%

        # ── Eval dataset ───────────────────────────────────────────────────
        eval_dataset_size=10,        # generate 10 synthetic test cases
        train_ratio=0.50,
        val_ratio=0.25,
        holdout_ratio=0.25,

        # ── Output ─────────────────────────────────────────────────────────
        output_dir=Path("./skill_evolver_output"),
        run_pytest=False,
    )

    print("Starting GEPA evolution...")
    print(f"  Skill     : git-review")
    print(f"  Iterations: {config.iterations}")
    print(f"  Models    : optimizer={config.optimizer_model}  eval={config.eval_model}")
    print()

    params: SkillEvolverParams = SkillEvolverParams(skill_name="git-review",
                                                    eval_source="synthetic", # generate dataset from skill using LLM
                                                    config=config,)

    try:
        metrics = evolve_single_skill(params)
    except FileNotFoundError as exc:
        print(f"Skill not found: {exc}")
        print("\nCreate a minimal skill first — see docstring above.")
        return

    print("\n── Results ─────────────────────────────────────────")
    print(f"  Baseline score : {metrics['baseline_score']:.4f}")
    print(f"  Evolved score  : {metrics['evolved_score']:.4f}")
    print(f"  Improvement    : {metrics['improvement']:+.4f}")
    print(f"  Elapsed        : {metrics['elapsed_seconds']:.1f}s")
    print(f"  Output dir     : {config.output_dir / 'git-review'}")
    print()

    # Constraint summary
    print("── Constraint checks ───────────────────────────────")
    for check in metrics["constraint_checks"]:
        status = "✓" if check["passed"] else "✗"
        print(f"  {status} {check['name']:20s}  {check['message']}")


if __name__ == "__main__":
    run_example_evolution()
