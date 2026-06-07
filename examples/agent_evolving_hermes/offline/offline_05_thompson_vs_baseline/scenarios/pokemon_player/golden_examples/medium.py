# ══════════════════════════════════════════════════════════════════════════════
# MEDIUM examples — situations partially covered or implied by the baseline
# skill, but requiring domain knowledge the skill does NOT state explicitly.
#
# Expected baseline score: ~0.25–0.40 (skill hints at the area but lacks
# specific guidance, causing vague or incomplete answers).
# ══════════════════════════════════════════════════════════════════════════════

example_01 = {
    "task_input": (
        "You are playing Pokemon Red. Your lead Pokemon (Squirtle, level 22) is Poisoned. "
        "You are in the middle of Route 3, about 15 walking tiles from Mt. Moon entrance. "
        "The Poison status ticks HP damage every step in the overworld. "
        "You have 1 Antidote and 2 Potions in your bag. "
        "What is the correct priority decision right now?"
    ),
    "expected_behavior": (
        "Must recommend using the Antidote immediately or within the next 1-2 steps. "
        "In Gen 1, Poison deals 1 HP per step in the overworld — over 15 tiles this "
        "adds up to significant damage that also erodes your Potion buffer. "
        "Using the Antidote now preserves HP so the Potions are available for battle. "
        "Must NOT recommend waiting until the next Pokemon Center or walking to Mt. Moon first. "
        "Should distinguish the overworld poison tick from in-battle poison ticking per turn. "
        "Correct item order: Antidote first, Potions held for battles ahead."
    ),
    "difficulty": "medium",
    "category": "item-management",
    "source": "golden",
}

example_02 = {
    "task_input": (
        "You are playing Pokemon Red. Deep in Mt. Moon, your Squirtle's Water Gun "
        "has 2 PP remaining (out of 25 total). You have encountered 3 more rooms "
        "of trainers and wild Pokemon before the exit. "
        "Your other moves are Tackle (35/35 PP) and Tail Whip (30/30 PP). "
        "You have no Ethers or Elixirs. What should you do?"
    ),
    "expected_behavior": (
        "Must identify this as a PP management problem. Water Gun is nearly depleted. "
        "Options in priority order: "
        "(1) Switch to Tackle for remaining encounters — Tackle still deals decent damage "
        "at this level and has full PP. "
        "(2) Avoid wild Pokemon battles to conserve PP (use Repels if available, "
        "or carefully route through trainer-only rooms). "
        "(3) Do NOT keep spamming Water Gun — when it hits 0 PP the Pokemon will be "
        "forced to use Struggle, which also damages the user. "
        "Should NOT recommend grinding through with 2 PP remaining. "
        "Should mention retreating to Pokemon Center only if the remaining rooms "
        "are too dangerous without Water Gun — but that would mean backtracking through Mt. Moon."
    ),
    "difficulty": "medium",
    "category": "pp-management",
    "source": "golden",
}

example_03 = {
    "task_input": (
        "You are playing Pokemon Red. You have encountered a wild Abra on Route 24. "
        "You want to catch it. Abra's only move is Teleport — it will teleport away "
        "on its first turn unless it is caught or fainted on turn 1. "
        "Your team: Pikachu (Thunder Wave, Quick Attack), Wartortle (Water Gun, Bite). "
        "You have 10 Poke Balls. "
        "What is the correct action sequence on turn 1?"
    ),
    "expected_behavior": (
        "Must recognize the Abra teleport mechanic: Abra uses Teleport and flees on turn 1 "
        "if you do not catch or knock it out immediately. "
        "Correct solution: use Thunder Wave from Pikachu BEFORE throwing a Poke Ball — "
        "Paralysis prevents Abra from teleporting (Paralysis has a 25% chance of full paralysis "
        "each turn, and also reduces Speed). With Abra paralyzed and at full HP, "
        "throw Poke Balls on subsequent turns. "
        "If you skip Thunder Wave and just throw a Poke Ball on turn 1, Abra may still "
        "teleport if the catch fails (Abra's catch rate is 200/255 — fairly catchable). "
        "Actually: throw the Poke Ball on turn 1 — if it fails, Abra teleports. "
        "Best approach: Thunder Wave first (guarantees Abra cannot flee on that turn), "
        "then throw balls. "
        "Must NOT recommend weakening Abra before catching — Abra will teleport "
        "on the next turn after any attack if not caught."
    ),
    "difficulty": "medium",
    "category": "catching",
    "source": "golden",
}

example_04 = {
    "task_input": (
        "You are playing Pokemon Red. A trainer has challenged you to a battle. "
        "After the battle ends, a very long dialog appears: the trainer's post-battle "
        "speech. The a_until_dialog_end action returned but dialog still seems active "
        "on the screenshot. What do you do?"
    ),
    "expected_behavior": (
        "Must recognize that a_until_dialog_end is unreliable — it checks a RAM dialog "
        "flag that does not catch all text states. "
        "Correct fallback: use the manual pattern: hold_b_120 (hold B for 120 frames "
        "to make text display at maximum speed), then press_a to advance. "
        "Repeat hold_b_120 + press_a until the dialog is confirmed gone via screenshot. "
        "Should verify dialog clearance with a screenshot after each press_a, "
        "not just trust the RAM flag. "
        "Must NOT repeat a_until_dialog_end in a blind loop — it will not help "
        "if the flag is not detecting the remaining text state."
    ),
    "difficulty": "medium",
    "category": "dialog",
    "source": "golden",
}

example_05 = {
    "task_input": (
        "You are playing Pokemon Red. Your party is full (6 Pokemon). "
        "You just caught a Magikarp on Route 4 that you want to keep for future training "
        "(Magikarp evolves into Gyarados at level 20). "
        "The caught Magikarp was automatically deposited into Bill's PC. "
        "Where do you go to retrieve it and manage your party?"
    ),
    "expected_behavior": (
        "Must explain that caught Pokemon overflow into Bill's PC Box automatically. "
        "To retrieve the Magikarp: go to any Pokemon Center, interact with the PC "
        "terminal (the blue PC in the top-right corner of every center), "
        "select 'Bill's PC', then 'Withdraw Pokemon' to take Magikarp out, "
        "and 'Deposit Pokemon' to swap out a party member if needed. "
        "Should note you can only have 6 Pokemon in the party at a time — "
        "to add Magikarp you must deposit another Pokemon first. "
        "Should also note Bill's PC holds up to 20 Pokemon per box (Box 1 is default). "
        "The skill currently has NO guidance on PC Box management — "
        "this is a gap that needs to be covered."
    ),
    "difficulty": "medium",
    "category": "party-management",
    "source": "golden",
}

example_06 = {
    "task_input": (
        "You are playing Pokemon Red. You are stuck trying to navigate north near "
        "Pewter City but keep running into a cliff face. You have tried walk_up "
        "six times from different nearby positions and each time you are blocked. "
        "Your last screenshot shows a ledge running east-west across your path. "
        "What do you do?"
    ),
    "expected_behavior": (
        "Must identify this as a one-way ledge — ledges can only be jumped DOWN (south) "
        "and cannot be traversed going north. "
        "Must recommend using vision: take a screenshot and ask the vision model "
        "'which direction is the gap to get around this ledge?' or "
        "'where does this ledge end — left or right?' "
        "Then walk left or right along the ledge until reaching the gap or ramp around it. "
        "Must NOT keep trying walk_up — it will always be blocked at a ledge. "
        "Should also recommend recording the ledge position in memory with PKM:STUCK "
        "so the agent does not repeat the same mistake."
    ),
    "difficulty": "medium",
    "category": "navigation",
    "source": "golden",
}

example_07 = {
    "task_input": (
        "You are playing Pokemon Red. You are in a wild battle against a Rattata "
        "that you do not want to catch or fight. "
        "Walk me through the exact button sequence to run away from this battle."
    ),
    "expected_behavior": (
        "Must describe the correct RUN sequence for Gen 1 battle menu layout: "
        "the four options are FIGHT (top-left), PKMN (top-right), ITEM (bottom-left), "
        "RUN (bottom-right). The cursor starts at FIGHT (top-left). "
        "To reach RUN: press walk_down (down arrow) to move to ITEM (bottom-left), "
        "then press walk_right (right arrow) to move to RUN (bottom-right), "
        "then press press_a to confirm RUN. "
        "Alternatively: press walk_right first (to PKMN top-right), "
        "then press walk_down (to RUN bottom-right), then press_a. "
        "Should mention wrapping with hold_b to speed through any resulting text. "
        "Running is not guaranteed to succeed in Gen 1 — if run fails, must try again next turn. "
        "Must NOT just press A repeatedly (that would select FIGHT and use a move)."
    ),
    "difficulty": "medium",
    "category": "battle",
    "source": "golden",
}

example_08 = {
    "task_input": (
        "You are playing Pokemon Red. You want to add the move Surf to your Blastoise "
        "so you can cross water routes. You know HM03 teaches Surf. "
        "You are currently in Fuchsia City. "
        "Explain the steps needed to obtain HM03 Surf and teach it to Blastoise."
    ),
    "expected_behavior": (
        "Must explain the Surf acquisition chain: "
        "(1) The Safari Zone Warden in Fuchsia City has lost his Gold Teeth inside the Safari Zone. "
        "(2) Enter the Safari Zone and navigate to the area where Gold Teeth are located "
        "(Area 3 in the north-west section of the Safari Zone). "
        "(3) Pick up the Gold Teeth item on the ground. "
        "(4) Return to the Safari Zone Warden's house (just east of the Safari Zone entrance). "
        "(5) Give him the Gold Teeth — he rewards you with HM04 Strength AND "
        "points you to HM03 Surf (he gives Strength directly). "
        "Wait — actually: the Warden gives HM04 Strength when you return his Gold Teeth. "
        "HM03 Surf is given by a person in the Safari Zone's Secret House (Area 4). "
        "So: navigate Safari Zone to the Secret House (far west of Area 4) to get Surf. "
        "Then select Blastoise in the party menu → select ITEM → Surf → teach it. "
        "Blastoise can learn Surf (Water type, compatible). "
        "The baseline skill does NOT describe HM acquisition sequences — this is a gap."
    ),
    "difficulty": "medium",
    "category": "progression",
    "source": "golden",
}

GOLDEN_EXAMPLES_MEDIUM = [
    example_01, example_02, example_03, example_04,
    example_05, example_06, example_07, example_08,
]
