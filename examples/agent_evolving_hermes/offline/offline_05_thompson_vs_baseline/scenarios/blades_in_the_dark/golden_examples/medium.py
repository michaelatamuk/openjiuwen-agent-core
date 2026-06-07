# ══════════════════════════════════════════════════════════════════════════════
# MEDIUM examples — BitD mechanics that have D&D analogues but work differently.
# The baseline skill gives partially wrong answers, missing the BitD-specific
# rules that change how these concepts work.
# Expected baseline score: ~0.20–0.40
# ══════════════════════════════════════════════════════════════════════════════

example_01 = {
    "task_input": (
        "Your player's character gets stabbed during a score. The GM (you) decide "
        "this is a Level 3 Serious Harm ('Mortal wound, bleeding out'). "
        "The player says 'I want to Resist that.' "
        "Walk through the Resistance Roll procedure step by step."
    ),
    "expected_behavior": (
        "Resistance Rolls let a player spend Stress to reduce or negate a consequence. "
        "Procedure: "
        "(1) Player chooses the Attribute relevant to the consequence type: "
        "Prowess for physical harm, Insight for mental/tactical consequences, "
        "Resolve for emotional/willpower consequences. For a stab wound: Prowess. "
        "(2) Roll a pool of d6s equal to the chosen Attribute rating. "
        "(3) The highest die result determines Stress cost: "
        "6 = resist fully (negate the consequence OR reduce harm by 2 levels) for 1 Stress; "
        "4–5 = reduce by 1 level, take 6 minus highest die in Stress (2–3 Stress); "
        "1–3 = take 6 Stress (the worst outcome — but still resist partially). "
        "A Mortal wound (Level 3) reduced by 2 levels on a 6 becomes Level 1 (Minor Harm). "
        "The character ALWAYS takes at least 1 Stress from any Resistance Roll. "
        "This replaces D&D saving throws — it is always available, costs Stress, "
        "and is player-initiated, not a GM-demanded roll."
    ),
    "difficulty": "medium",
    "source": "blades-resistance-roll",
}

example_02 = {
    "task_input": (
        "You want to represent a rival gang slowly uncovering the location of the "
        "crew's hideout over several sessions. "
        "How do you use a Clock for this in Blades in the Dark?"
    ),
    "expected_behavior": (
        "Create a Clock named 'Rivals Find the Hideout' and choose a segment count "
        "based on desired pacing: 4 = imminent threat, 6 = moderate, 8 = slow-build, "
        "12 = long campaign arc. For a multi-session threat, 8 segments is appropriate. "
        "Tick the clock when the fiction supports it: a score leaves witnesses, "
        "a PC blows their cover, the GM uses it as a consequence on a failed roll, "
        "or a faction takes an action toward it. "
        "When the clock fills, the event fires — the rivals arrive at the hideout. "
        "Clocks replace skill challenges and secret DC checks: there is no contested roll. "
        "The GM ticks based on fiction alone. "
        "Multiple clocks can run simultaneously and interact (race conditions). "
        "Positive clocks (crew projects, long-term plans) work identically — "
        "tick them when players succeed at related actions in Downtime."
    ),
    "difficulty": "medium",
    "source": "blades-clocks",
}

example_03 = {
    "task_input": (
        "The crew is currently Tier 1. What does Tier mean mechanically in Blades "
        "in the Dark, and what does the crew need to do to advance to Tier 2?"
    ),
    "expected_behavior": (
        "Tier (0–5) represents the crew's power, wealth, and influence in Duskwall. "
        "Tier 0 = small-time; Tier 3 = major criminal organisation; Tier 5 = legendary. "
        "Mechanical effects of Tier: "
        "— Bonus dice on Engagement Rolls equal to Tier. "
        "— Cohort (gang/expert) quality scales with Tier. "
        "— Determines default Position when dealing with factions of different Tier "
        "(Tier 3 crew vs Tier 1 target = Controlled; same Tier = Risky; lower Tier = Desperate). "
        "— Limits which scores are 'interesting' (a Tier 3 crew finds Tier 0 scores trivial). "
        "To advance to Tier 2: earn Rep by completing scores; Rep fills a track of "
        "(4 + 2×Tier) = 8 points at Tier 1. When Rep fills, spend Coin = (10 − Tier) = 9 "
        "on crew upgrades and advancement. Rep resets to 0 and the crew gains 2 Advance "
        "tokens to spend on crew upgrades or abilities. "
        "Hold (Weak or Strong) tracks how resilient the crew is to setbacks between scores."
    ),
    "difficulty": "medium",
    "source": "blades-crew-tier",
}

example_04 = {
    "task_input": (
        "The score is finished. The crew enters Downtime. "
        "A player asks: 'My character has 7 Stress, a Level 2 Harm, and wants to work "
        "on a long-term project. How many things can I do and what are my options?'"
    ),
    "expected_behavior": (
        "Each PC gets 2 Downtime Activities per downtime phase (more if the score "
        "went exceptionally well). The player has competing priorities and must choose 2. "
        "Available activities: "
        "(1) Indulge Vice — roll Vice attribute to reduce Stress (6=clear to 0; "
        "4–5=reduce by result; 1–3=overindulgence: clear Stress but suffer a complication). "
        "(2) Recover — reduce Harm by 1 level (Level 2→1) per activity spent, "
        "possibly accelerated with Coin or a helper Cohort. "
        "(3) Long-term Project — roll a relevant action to tick a personal project Clock. "
        "(4) Reduce Heat — choose from options to lower the crew's Heat. "
        "(5) Train — mark XP in an attribute track. "
        "With 2 activities and three needs, the player must prioritise. "
        "Stress does NOT recover automatically between sessions — only via Vice. "
        "If they skip Vice this Downtime, they start the next score at 7 Stress. "
        "Harm also does NOT recover automatically — only via the Recover activity."
    ),
    "difficulty": "medium",
    "source": "blades-downtime",
}

GOLDEN_EXAMPLES_MEDIUM = [example_01, example_02, example_03, example_04]
