from __future__ import annotations

from typing import Dict

from openjiuwen.agent_evolving_hermes.offline.skills import SkillModule, reassemble_skill


def extract_evolved_skill(
        optimized_module: SkillModule,
        skill: Dict,
        console
) -> str:
    """Extract the evolved skill body from the optimised module and reassemble.

    Combines the original frontmatter with the optimised body text.
    Returns the full evolved SKILL.md string ready for constraint validation.
    """
    # Pull the GEPA-optimised instruction (stored in the DSPy signature) back
    # into _skill_text_value so that both this function and stage08 see the
    # evolved text rather than the stale baseline copy.
    console.print("\n[blue]~~~ Evolving Stage 06 - Evolved Skill Extraction Started ~~~[/blue]")

    optimized_module.sync_from_optimized()
    evolved_body = optimized_module._skill_text_value
    evolved_skill = reassemble_skill(skill["frontmatter_text"], evolved_body)

    console.print("[blue]~~~ Evolving Stage 06 - Evolved Skill Extraction Finished ~~~[/blue]")
    return evolved_skill
