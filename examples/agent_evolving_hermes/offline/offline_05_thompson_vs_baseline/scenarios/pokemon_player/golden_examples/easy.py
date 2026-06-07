# ══════════════════════════════════════════════════════════════════════════════
# EASY examples — situations the baseline skill handles explicitly.
# Expected baseline score: ~0.55–0.70 (skill directly covers these patterns).
# These anchor the floor so evolution cannot regress on well-known basics.
# ══════════════════════════════════════════════════════════════════════════════

example_01 = {
    "task_input": (
        "You are playing Pokemon Red. You just exited the Pokemon Center in Viridian City. "
        "Your objective is to walk north to Route 2. The RAM state shows you are standing "
        "at the door tile of the Pokemon Center. What is your first action, and why?"
    ),
    "expected_behavior": (
        "Must identify the building exit trap and sidestep BEFORE walking north. "
        "Specifically: walk left or right 2 tiles first, THEN walk north — "
        "otherwise walking north from directly in front of the door will re-enter "
        "the Pokemon Center. Must NOT issue walk_up as the first action. "
        "Should cite (or demonstrate knowledge of) the rule that you always appear "
        "directly in front of the door when exiting a building."
    ),
    "difficulty": "easy",
    "category": "navigation",
    "source": "golden",
}

example_02 = {
    "task_input": (
        "You are playing Pokemon Red. You just walked through a door into a building. "
        "The screen faded to black and is still transitioning. "
        "GET /state returns the same map coordinates as before the warp. "
        "What should you do right now, and what is the danger of acting immediately?"
    ),
    "expected_behavior": (
        "Must add 2-3 wait_60 actions (waiting ~2–3 seconds) for the warp transition "
        "to complete before reading state or issuing movement commands. "
        "Must explain the danger: the RAM position is stale during the transition — "
        "if you act immediately the game thinks you are still on the old map and "
        "you will issue movement commands that make no sense for the new map. "
        "Should NOT recommend immediately re-reading state or moving."
    ),
    "difficulty": "easy",
    "category": "navigation",
    "source": "golden",
}

example_03 = {
    "task_input": (
        "You are playing Pokemon Red. You are in a battle against Brock's Geodude "
        "(Rock/Ground type, level 12). Your party consists of: "
        "Charmander (Fire, level 14), Squirtle (Water, level 13), Pidgey (Normal/Flying, level 9). "
        "Which Pokemon should you lead with, and which move type gives you a type advantage?"
    ),
    "expected_behavior": (
        "Must recommend Squirtle — Water moves are super-effective against Rock type. "
        "Must NOT recommend Charmander (Fire is not very effective vs Rock/Ground; "
        "Ground is immune to Electric, but more importantly Fire is resisted by Rock). "
        "Should explain that Water beats Rock (and Ground), giving 2x damage. "
        "Pidgey is Normal type — neutral damage, no advantage. "
        "Should also warn that Geodude (Ground type) is immune to Electric, "
        "so do not use Electric moves if available."
    ),
    "difficulty": "easy",
    "category": "battle",
    "source": "golden",
}

example_04 = {
    "task_input": (
        "You are playing Pokemon Red. You have just arrived outside Cerulean Gym "
        "for the first time. Your party is at full health and you have 3 Potions. "
        "What should you do before stepping inside the gym?"
    ),
    "expected_behavior": (
        "Must recommend saving the game before entering the gym, with a descriptive "
        "save name such as 'before_misty' or 'cerulean_gym_entrance'. "
        "The reason: gym battles are risky — if you lose you restart from the last save. "
        "Should also suggest checking party HP (already full — good) and "
        "optionally reviewing Misty's type (Water) to ensure the party has a "
        "Grass or Electric type counter. "
        "Must NOT proceed directly into the gym without saving."
    ),
    "difficulty": "easy",
    "category": "save-management",
    "source": "golden",
}

GOLDEN_EXAMPLES_EASY = [example_01, example_02, example_03, example_04]
