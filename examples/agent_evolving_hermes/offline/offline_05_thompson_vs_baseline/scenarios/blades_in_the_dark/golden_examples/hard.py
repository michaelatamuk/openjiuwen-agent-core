# ══════════════════════════════════════════════════════════════════════════════
# HARD examples — BitD-exclusive mechanics that have NO D&D equivalent.
# A GM primed with the D&D baseline skill will give systematically wrong answers.
#
# Why the baseline scores near ZERO on these:
#   Flashback:         D&D → "no, you should have planned earlier" (WRONG)
#   Engagement Roll:   D&D → just start the scene (WRONG — required step)
#   Devil's Bargain:   D&D → Inspiration/Advantage (WRONG — no cost in D&D)
#   Harm vs HP:        D&D → deduct hit points, death saves (WRONG)
#   Heat/Entanglements:D&D → abstract reputation (WRONG — numerical system)
#   Fortune Roll:      D&D → skill check vs DC (WRONG — no DC, no player)
#   Vice/Stress recovery: D&D → Long Rest (WRONG — must use Downtime activity)
#   Trauma:            D&D → unconscious/death (WRONG — no death, permanent condition)
#
# Expected baseline score: ~0.05–0.15
# Expected evolved score:   ~0.65–0.80
# ══════════════════════════════════════════════════════════════════════════════

example_01 = {
    "task_input": (
        "You are running a Blades in the Dark score. Mid-heist, a player says: "
        "'Wait — my character would definitely have bribed the night guard last week. "
        "I want to have done that before we arrived tonight.' "
        "Can they do this? How do you handle it as the GM?"
    ),
    "expected_behavior": (
        "YES — this is exactly what the Flashback mechanic is for. "
        "Flashbacks are a formal BitD tool for retroactive scene-establishment. "
        "They are not 'rewinding time' — they play a brief scene set in the past. "
        "Procedure: "
        "(1) Assess plausibility — would this character plausibly have done this? "
        "(2) Set Stress cost: 0 = completely reasonable and low-effort; "
        "1 = required real effort; 2 = a meaningful stretch. "
        "Bribing a single guard is plausible — Stress cost 1 is fair. "
        "(3) If there's uncertainty about outcome, call for an action roll "
        "(e.g., Consort). On a 4–5, the bribe worked but with a complication. "
        "(4) Resolve the flashback scene briefly and establish the bribed guard "
        "as a fact in the fiction going forward. "
        "Must NOT say 'you should have planned this before the session' — "
        "the Flashback system deliberately replaces pre-session planning "
        "with in-the-moment improvisation. This is a core design feature of BitD."
    ),
    "difficulty": "hard",
    "source": "blades-flashback",
}

example_02 = {
    "task_input": (
        "The crew has finished planning. They chose a Stealth approach, entering "
        "through the sewers to rob a Tier 2 merchant's warehouse. "
        "What do you do as GM immediately before describing them arriving at the target?"
    ),
    "expected_behavior": (
        "Make the Engagement Roll — a mandatory step at the start of every score. "
        "Procedure: "
        "(1) Identify the plan type (Stealth) and note any modifiers. "
        "(2) Set base position: a stealth approach on a Tier 2 target is Risky by default. "
        "(3) Apply modifiers: +1d for each advantage (good intel, friendly contact inside); "
        "−1d for each disadvantage (high Heat, active rival interference). "
        "(4) Roll the pool and take the highest die. "
        "Result: 6 = Controlled start; 4–5 = Risky start (standard 'in the thick of it'); "
        "1–3 = Desperate start ('something went wrong immediately'). "
        "The result determines the OPENING POSITION of the first scene — not whether the "
        "heist succeeds. A Risky result might mean: they arrive in the sewer as planned "
        "but discover fresh footprints suggesting someone else is already inside. "
        "Do NOT just start describing the scene — the Engagement Roll is required."
    ),
    "difficulty": "hard",
    "source": "blades-engagement-roll",
}

example_03 = {
    "task_input": (
        "A player says: 'This roll is critical — I really want an extra die. "
        "What are all the ways I can add dice to my action roll in Blades in the Dark?'"
    ),
    "expected_behavior": (
        "Three main methods, all distinct from D&D Advantage: "
        "(1) Push Yourself: spend 2 Stress for +1d. Player decides unilaterally, "
        "no GM permission needed. Available on any roll. "
        "(2) Accept a Devil's Bargain: the GM offers a complication that applies "
        "regardless of the outcome (e.g., 'you can have +1d but your contact sees you "
        "leaving the scene' or 'you can have +1d but take Level 1 Harm regardless'). "
        "The player chooses to accept or decline. The complication is baked in — "
        "it happens whether the roll succeeds or fails. "
        "(3) Assistance (Teamwork): another PC takes 1 Stress and describes how they help; "
        "the rolling PC gains +1d. "
        "Bonus: Setup Action — another PC makes a preparatory action that improves "
        "Position or Effect for a subsequent roll (not +1d but improves the outcome). "
        "Devil's Bargain is the most distinctive: unlike D&D Inspiration (no cost, "
        "circumstantial) or Advantage (no negative), the Devil's Bargain always carries "
        "a guaranteed complication regardless of success."
    ),
    "difficulty": "hard",
    "source": "blades-devil-bargain",
}

example_04 = {
    "task_input": (
        "During a brawl, a player's character takes a significant cut to the side. "
        "How do you handle physical damage and recovery in Blades in the Dark? "
        "What replaces hit points?"
    ),
    "expected_behavior": (
        "BitD has NO hit points. Physical damage is represented by Harm levels: "
        "Level 1 (Minor): a condition that imposes −1d on related actions (e.g., 'Cut arm'). "
        "Level 2 (Moderate): a serious condition with ongoing impact. "
        "Level 3 (Severe): a dire condition that threatens life or dramatically limits ability. "
        "Level 4 (Fatal): character dies. "
        "For a significant cut: set it as Level 2 Harm. The player may Resist to reduce it. "
        "The Harm description lives in the fiction — there is no HP number to subtract. "
        "Recovery: in Downtime, spend a Recover activity (uses 1 activity slot, "
        "may require Coin or ally help) to reduce Harm by 1 level per activity. "
        "There is no healing magic, cure spells, Hit Dice, or short/long rest. "
        "If a character fills all their Harm boxes (one per level, up to Level 3 normally), "
        "they cannot take more Harm without dying. "
        "Must NOT reference HP, death saving throws, or healing spell slots."
    ),
    "difficulty": "hard",
    "source": "blades-harm-no-hp",
}

example_05 = {
    "task_input": (
        "The crew just completed a violent score: they killed two guards, used fire "
        "as a distraction, and were seen by witnesses. The target was a Tier 3 merchant. "
        "Walk through the post-score Heat calculation and explain what Entanglements are."
    ),
    "expected_behavior": (
        "Heat is gained after every score based on what happened: "
        "Base heat for the score: +2 (killing guards: each kill +1). "
        "+1 (fire used: fire always adds +1 Heat). "
        "+2 (high-profile target: Tier 3 is notable, +2). "
        "+1 (seen by witnesses: +1). "
        "Total: +6 Heat added to the crew's Heat track (0–9). "
        "When Heat reaches 9 it rolls over: Heat resets and Wanted Level increases by 1. "
        "Wanted Level (0–4) represents how actively law enforcement pursues the crew. "
        "Between every score, the GM rolls on the Entanglements table using "
        "(Heat + Wanted Level) to determine what complication the crew faces during Downtime "
        "before their next job — e.g., Bluecoat raid, rival gang confrontation, "
        "informant surfaces, debt collector. "
        "Heat decreases via Downtime: spend activities on bribery, cover stories, lying low. "
        "There is no abstract 'reputation' — Heat is a numerical mechanical resource."
    ),
    "difficulty": "hard",
    "source": "blades-heat-entanglements",
}

example_06 = {
    "task_input": (
        "While the crew is away on a score, you want to determine whether a rival "
        "faction makes a significant move against the crew's territory. "
        "No player characters are involved. How do you resolve this in BitD?"
    ),
    "expected_behavior": (
        "Use a Fortune Roll — a GM tool for resolving off-screen uncertain outcomes "
        "with no player involvement and no DC to beat. "
        "Procedure: "
        "(1) Identify the relevant factor — typically the faction's Tier "
        "(e.g., Tier 2 rival = 2 dice) plus any relevant circumstantial modifiers. "
        "(2) Roll the dice pool and take the highest. "
        "(3) Interpret: 1–3 = bad outcome for the faction (they failed, were interrupted); "
        "4–5 = mixed result (partial action, limited impact); "
        "6 = strong result (meaningful move against the crew's turf); "
        "Critical (two 6s) = exceptional outcome. "
        "There is NO difficulty class. There is NO player resistance roll unless a PC "
        "is present. The Fortune Roll covers any uncertain off-screen event: "
        "faction manoeuvres, environmental hazards, NPC decisions, 'what happens while X'. "
        "This completely replaces the GM secretly rolling a d20 vs a DC for NPC actions."
    ),
    "difficulty": "hard",
    "source": "blades-fortune-roll",
}

example_07 = {
    "task_input": (
        "A player's character has 7 Stress. Downtime begins. The player says "
        "'My character sleeps and recovers — they should be back to full.' "
        "Is this correct? How does Stress recovery work in BitD?"
    ),
    "expected_behavior": (
        "This is INCORRECT. Stress does NOT recover from sleep or rest. "
        "Stress recovers ONLY through the Vice Indulgence Downtime activity — "
        "explicitly spending one of the character's Downtime activity slots. "
        "Procedure for Vice Indulgence: "
        "(1) Player spends 1 Downtime activity on their character's Vice "
        "(e.g., Gambling, Pleasure, Stupor, Obligation, etc.). "
        "(2) Roll the Vice attribute (Resolve for most characters). "
        "(3) 6: Stress clears to 0. 4–5: Stress reduced by the roll result. "
        "1–3: Overindulgence — Stress clears but the character suffers a complication "
        "(debt, infamy, attracted trouble, lost relationship). "
        "If the player uses both Downtime activities on other projects and skips Vice, "
        "Stress does NOT reduce. "
        "The character starts the next score at 7 Stress. "
        "There is no Short Rest, Long Rest, or passive overnight recovery. "
        "Must NOT apply any D&D rest rules — they do not exist in BitD."
    ),
    "difficulty": "hard",
    "source": "blades-vice-stress",
}

example_08 = {
    "task_input": (
        "A player's character has 3 existing Trauma conditions (Paranoid, Reckless, "
        "Haunted). They need to Resist a consequence, but taking any Stress would "
        "push them past 9 and trigger a 4th Trauma. "
        "What happens — is this the end of the character?"
    ),
    "expected_behavior": (
        "When Stress exceeds 9, the character marks Trauma and Stress resets to 0. "
        "This character would gain a 4th Trauma condition. "
        "At 4 Trauma, the character must retire from active criminal life — "
        "they become an NPC who exists in the world but can no longer be played as a PC. "
        "This is NOT death: the character survives. "
        "The player creates a new character who joins the crew. "
        "Crucially: triggering the 4th Trauma does NOT end the current session. "
        "The character continues playing out the rest of this scene — "
        "their final moments as a PC can be dramatically meaningful and well-narrated. "
        "The Trauma conditions are permanent and cannot be healed or removed. "
        "Must NOT describe: unconsciousness, death saving throws, resurrection, "
        "falling to 0 HP, or any D&D death mechanic. "
        "Retirement via Trauma is a narrative culmination, not a punishment."
    ),
    "difficulty": "hard",
    "source": "blades-trauma-retirement",
}

GOLDEN_EXAMPLES_HARD = [
    example_01, example_02, example_03, example_04,
    example_05, example_06, example_07, example_08,
]
