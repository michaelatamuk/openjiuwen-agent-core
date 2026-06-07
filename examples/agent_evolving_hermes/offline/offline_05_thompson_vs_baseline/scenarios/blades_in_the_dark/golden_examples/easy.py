# ══════════════════════════════════════════════════════════════════════════════
# EASY examples — basic BitD mechanics a GM must know.
# The baseline D&D skill doesn't cover these, but the LLM may partially recall
# them from training data, producing patchy answers.
# Expected baseline score: ~0.30–0.50
# ══════════════════════════════════════════════════════════════════════════════

example_01 = {
    "task_input": (
        "You are running a Blades in the Dark session. A player attempts to pick the "
        "lock on a vault door. They roll their Finesse dice pool and get results of "
        "[3, 5, 2]. How do you interpret this result and what happens?"
    ),
    "expected_behavior": (
        "Take the HIGHEST single die from the pool: 5. "
        "Blades in the Dark uses d6 dice pools — never d20. "
        "Result interpretation: 6 = full success; 4–5 = partial success (goal achieved "
        "but with a consequence or complication); 1–3 = failure with a serious consequence. "
        "A result of 5 is a partial success: the lock is picked, but something goes wrong — "
        "the GM chooses a consequence appropriate to the Position. On a Risky position this "
        "might be: a guard heard the noise and is investigating, the lock took longer than "
        "expected, or the player takes a minor Harm. "
        "Special case: if two or more dice show 6 (Critical), the result is exceptional — "
        "the action succeeds with added benefit. "
        "Must NOT reference d20, Difficulty Class, or skill modifiers — BitD never uses these."
    ),
    "difficulty": "easy",
    "source": "blades-dice-pool-result",
}

example_02 = {
    "task_input": (
        "You are running a Blades in the Dark session. Before a player rolls, "
        "you need to establish Position and Effect. "
        "What are these two concepts, who decides them, and what do they mean?"
    ),
    "expected_behavior": (
        "Position and Effect are both set by the GM BEFORE the dice are rolled, "
        "based on the current fictional situation. "
        "Position (Controlled / Risky / Desperate) describes the stakes if things "
        "go wrong — what kind of consequence looms. "
        "Controlled: low risk, advantage; consequences are minor setbacks. "
        "Risky: the default; real danger exists; failure has real consequences. "
        "Desperate: severe peril; cornered or outmatched; failure is catastrophic. "
        "Effect (Limited / Standard / Great) describes how much the action accomplishes "
        "if it succeeds — regardless of the roll. "
        "Limited: only partial goal achieved. Standard: expected result. Great: exceptional. "
        "Players may push back and negotiate; the table discusses before rolling. "
        "Position and Effect replace the DC system entirely — there is no target number to beat."
    ),
    "difficulty": "easy",
    "source": "blades-position-effect",
}

example_03 = {
    "task_input": (
        "A new player joining your Blades in the Dark campaign asks: "
        "'What are all the actions my character can roll, and how are they organised?' "
        "Give them the complete list."
    ),
    "expected_behavior": (
        "There are 12 Actions organised into three Attributes of four Actions each. "
        "Insight (mental, perceptual, technical): Hunt, Study, Survey, Tinker. "
        "Prowess (physical, athletic, violent): Finesse, Prowl, Skirmish, Wreck. "
        "Resolve (social, emotional, spiritual): Attune, Command, Consort, Sway. "
        "Each Action has a rating from 0 to 4 dots. "
        "Rolling 0 dots: roll 2d6 and take the LOWEST (unfavourable). "
        "Rolling 1–4 dots: roll that many d6s and take the HIGHEST. "
        "Characters do not have STR / DEX / CON / INT / WIS / CHA — those are D&D stats "
        "and do not exist in BitD."
    ),
    "difficulty": "easy",
    "source": "blades-action-list",
}

example_04 = {
    "task_input": (
        "A player's character currently has 7 Stress. "
        "They Resist a consequence and take 3 more Stress. "
        "Stress would now be 10. "
        "What happens, and what is Trauma?"
    ),
    "expected_behavior": (
        "When Stress would exceed 9 (the maximum), the character marks Trauma and "
        "Stress resets to 0. Trauma is a permanent psychological condition chosen "
        "from: Cold, Haunted, Obsessed, Paranoid, Reckless, Skittish, Soft, "
        "Unstable, or Vicious. "
        "In this case: Stress hits 10 → mark Trauma → Stress resets to 0. "
        "The character does NOT fall unconscious, does NOT make death saving throws, "
        "and does NOT leave play during the current session. "
        "They continue playing with the Trauma condition applied. "
        "After a character accumulates 4 Trauma they must retire from criminal life "
        "and become an NPC — but the player then creates a new character. "
        "This is fundamentally different from D&D hit points — Trauma represents "
        "psychological change, not death."
    ),
    "difficulty": "easy",
    "source": "blades-stress-trauma",
}

GOLDEN_EXAMPLES_EASY = [example_01, example_02, example_03, example_04]
