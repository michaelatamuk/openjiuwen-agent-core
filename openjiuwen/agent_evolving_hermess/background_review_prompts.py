# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Background review prompt texts.

Faithful translations of Hermess's three review prompts into the Jiuwen context.
"""
from __future__ import annotations

MEMORY_REVIEW_PROMPT = """\
Review the conversation above and consider saving to memory if appropriate.

Focus on:
1. Has the user revealed things about themselves — their persona, desires,
   preferences, or personal details worth remembering?
2. Has the user expressed expectations about how you should behave, their work
   style, or ways they want you to operate?

If something stands out, save it using the memory_write tool.
If nothing is worth saving, just say 'Nothing to save.' and stop.
"""

SKILL_REVIEW_PROMPT = """\
Review the conversation and decide whether any skill should be created, updated, \
or patched. You have the full conversation history above.

━━━  PREFERENCE ORDER FOR UPDATES  ━━━
1. UPDATE the skill that was loaded and used during this conversation.
2. UPDATE the umbrella skill whose instructions govern this area of work.
3. ADD a supporting reference file to an existing skill.
4. CREATE a new narrow skill only if no existing skill fits.

━━━  SIGNALS TO LOOK FOR  ━━━
Any one of the following warrants action:

• The user corrected your style, tone, format, legibility, or verbosity.
  Frustration signals such as "stop doing X", "this is too verbose",
  "don't format like this", "just give me the answer", "you always do Y
  and I hate it", or an explicit "remember this" are FIRST-CLASS skill
  signals — act on them immediately.

• The user corrected your workflow, approach, or the sequence of steps you
  took. Encode the correction directly in the skill that governs that work.

• A non-trivial technique, fix, or workaround emerged during this session
  that would help future sessions — capture it as a Troubleshooting or
  Technique note inside the relevant skill.

• A skill that was used during this conversation is wrong, outdated, or
  missing important context — patch it now.

• Two loaded skills cover the same territory — note this for consolidation
  (add a comment in references/, do NOT delete either skill now).

━━━  WHAT NOT TO CAPTURE  ━━━
• Environment-dependent failures (missing binaries, OS-specific setup errors)
• Negative claims about tools ("the browser tool doesn't work here")
• Transient session errors that resolved on retry
• One-off task narratives that will not recur
• Information already present in an existing skill

━━━  PROTECTED SKILLS  ━━━
Do NOT edit skills whose frontmatter contains `immutable: true`.
These are bundled or hub-installed skills.

━━━  OUTPUT FORMAT  ━━━
Call skill_write, skill_patch, or skill_create as many times as needed.
If nothing warrants a change, say 'No skill changes needed.' and stop.
"""

COMBINED_REVIEW_PROMPT = """\
Review the conversation above for two things: memory-worthy user information \
and skill improvements.

━━━  MEMORY  ━━━
Save a memory entry if the user revealed durable preferences, personal details,
working style, or explicit expectations about your behaviour.
Use memory_write(target="memory", ...) or memory_write(target="user", ...).

━━━  SKILLS  ━━━
""" + SKILL_REVIEW_PROMPT.split("━━━  SIGNALS TO LOOK FOR")[1].split("━━━  OUTPUT FORMAT")[0] + """
━━━  OUTPUT FORMAT  ━━━
Call memory_write and/or skill_write/skill_patch/skill_create as needed.
If nothing needs updating, say 'No changes needed.' and stop.
"""


def select_prompt(mode: "ReviewMode") -> str:  # noqa: F821
    """Return the correct prompt string for the given ReviewMode."""
    from openjiuwen.agent_evolving_hermess.types import ReviewMode
    if mode == ReviewMode.MEMORY_ONLY:
        return MEMORY_REVIEW_PROMPT
    if mode == ReviewMode.SKILLS_ONLY:
        return SKILL_REVIEW_PROMPT
    return COMBINED_REVIEW_PROMPT
