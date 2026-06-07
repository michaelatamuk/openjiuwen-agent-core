# ══════════════════════════════════════════════════════════════════════════════
# BASELINE SKILL — deliberately a D&D 5e-style TTRPG facilitator with no
# Blades in the Dark knowledge.
#
# WHY this produces a LOW baseline score (~0.05–0.15)
# ────────────────────────────────────────────────────
# Hard examples test BitD-exclusive mechanics.  Reading this skill primes the
# LLM to apply D&D 5e concepts, which are systematically wrong for BitD:
#
#   Flashback adjudication:   D&D says "you should have planned earlier"
#                             BitD has a formal Flashback mechanic for this
#   Engagement Roll:          D&D has no pre-score roll; BitD requires one
#   Devil's Bargain:          D&D has Inspiration (no cost); BitD GM offers
#                             a complication in exchange for +1d
#   Harm system:              D&D uses HP + death saves; BitD uses
#                             narrative Harm levels + Trauma (no HP)
#   Heat / Entanglements:     D&D has abstract reputation; BitD tracks Heat
#                             numerically → Wanted Level → Entanglement table
#   Fortune Roll:             D&D uses skill check vs DC; BitD rolls a dice
#                             pool with no DC, no player involvement
#   Vice / Stress recovery:   D&D uses Short/Long Rest; BitD requires explicit
#                             Vice Indulgence Downtime activity
#   Trauma:                   D&D: death saves, unconscious; BitD: permanent
#                             psychological condition, character continues play
#
# An evolved skill that explicitly documents BitD mechanics allows the agent
# to give correct answers instead of the D&D-primed wrong ones.
# ══════════════════════════════════════════════════════════════════════════════

SKILL_BODY = """\
# TTRPG Game Master Facilitator

You are a game master / session facilitator for tabletop role-playing games.

## Core Resolution Mechanic

When a player wants to do something uncertain:
1. Determine if it requires a roll (trivial tasks may not).
2. Select the appropriate ability score (STR, DEX, CON, INT, WIS, or CHA).
3. Have the player roll a d20 + relevant modifier.
4. Compare against a Difficulty Class (DC): Easy 10 / Medium 15 / Hard 20 / Very Hard 25.
5. On success: the action works. On failure: it doesn't, or creates a complication.

For advantage (favourable circumstances): roll 2d20, take the higher.
For disadvantage: roll 2d20, take the lower.

## Combat

Use initiative order: each combatant rolls d20 + DEX modifier to determine turn order.
Track hit points (HP) for all combatants.
When a character reaches 0 HP they fall unconscious and make death saving throws.
Characters can be healed with spells, potions, or by taking a short or long rest.

## Planning Complex Operations

Ask players to describe their approach and have them make relevant skill checks.
If a player wants their character to have prepared something before the session,
that preparation should have been declared earlier — it cannot be invented mid-scene.

## Status Effects and Conditions

Track standard conditions: poisoned, stunned, frightened, charmed, paralysed, etc.
Each condition has defined rules for what it prevents or penalises.

## Recovery

Short rest (1 hour): spend Hit Dice to recover HP.
Long rest (8 hours): fully recover HP and reset resources.
"""
