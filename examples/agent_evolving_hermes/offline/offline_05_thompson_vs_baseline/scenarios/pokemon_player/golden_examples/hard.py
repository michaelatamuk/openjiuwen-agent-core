# ══════════════════════════════════════════════════════════════════════════════
# HARD examples — operational/procedural questions requiring SPECIFIC VALUES
# from the pokemon-player skill (API endpoints, action names, paths, port,
# memory prefixes, timing rules).  These CANNOT be answered correctly from
# general LLM training data; only an agent that has read the skill can answer.
#
# Why the baseline scores near ZERO on these:
#   - The LLM invents plausible-sounding but WRONG API endpoints, action names,
#     port numbers, and path names based on general patterns
#   - Generic answers like "take a screenshot" or "save the game" score ~0.1
#     because the judge looks for the exact values documented in the skill
#   - An evolved skill that restates or clarifies these concrete details
#     allows the agent to reproduce them faithfully → score ~0.80–0.95
#
# Expected baseline score: ~0.05–0.20
# Expected evolved score:   ~0.70–0.90
# ══════════════════════════════════════════════════════════════════════════════

example_01 = {
    "task_input": (
        "You are a Pokemon-playing agent. Describe the COMPLETE observation step "
        "at the start of every gameplay turn: list every API call you must make, "
        "in order, including the exact endpoint, any file paths used, "
        "and why each call is necessary."
    ),
    "expected_behavior": (
        "Must name both calls in order: "
        "(1) GET /state — returns structured RAM data: map coordinates, party HP, "
        "whether a battle or dialog is active. "
        "(2) GET /screenshot — captures the current screen frame; must be saved "
        "to /tmp/pokemon.png and then passed to vision_analyze. "
        "Must explain WHY both are required: RAM state gives numbers "
        "(position, HP) but cannot see ledges, fences, signs, or NPC positions — "
        "only vision provides spatial context. "
        "Must NOT omit either call or give a different path than /tmp/pokemon.png."
    ),
    "difficulty": "hard",
    "category": "api-procedure",
    "source": "golden",
}

example_02 = {
    "task_input": (
        "You are a Pokemon-playing agent. Your character just walked into a building "
        "and the screen is fading to black (warp transition). "
        "GET /state still shows the old map coordinates. "
        "What is the EXACT action payload you send next, and why must you do this "
        "before issuing any movement commands?"
    ),
    "expected_behavior": (
        "Must send a POST /action call containing 2–3 wait_60 actions "
        "(e.g. [wait_60, wait_60, wait_60]) before any movement. "
        "Must explain: wait_60 pauses for approximately 1 second (60 frames). "
        "During the warp transition the RAM position is stale — "
        "it still reports the old map location. "
        "Issuing walk_up/down/left/right immediately would execute on the old "
        "coordinate system and produce wrong movement in the new map. "
        "Only after waiting for the transition to complete does GET /state "
        "return valid coordinates for the new map. "
        "Must use the exact action name 'wait_60' (not 'wait', 'delay', or 'pause')."
    ),
    "difficulty": "hard",
    "category": "api-procedure",
    "source": "golden",
}

example_03 = {
    "task_input": (
        "You are a Pokemon-playing agent. A long NPC dialog is on screen. "
        "You tried sending a_until_dialog_end but dialog is still active "
        "according to the screenshot. "
        "What is the correct fallback action sequence to clear dialog reliably? "
        "Give the exact action names in order and explain why the first approach failed."
    ),
    "expected_behavior": (
        "Must prescribe the manual fallback: send [hold_b_120, press_a] and repeat "
        "until a screenshot confirms the dialog is gone. "
        "hold_b_120 holds the B button for 120 frames, which forces Gen 1 text "
        "to render at maximum speed. "
        "press_a then advances past the current dialog line. "
        "Must explain why a_until_dialog_end failed: it monitors a RAM dialog flag "
        "that does NOT capture all possible dialog states in Gen 1 — "
        "some text states are invisible to the flag. "
        "The screenshot must be used after each press_a to confirm clearance, "
        "not just the RAM flag. "
        "Must use exact action names: hold_b_120 and press_a "
        "(not 'hold_B', 'b_button', 'confirm', or 'advance')."
    ),
    "difficulty": "hard",
    "category": "api-procedure",
    "source": "golden",
}

example_04 = {
    "task_input": (
        "You are a Pokemon-playing agent. You have just reached Vermilion City "
        "and want to save before boarding the S.S. Anne. "
        "What is the exact API call to make, what format should the save name follow, "
        "and give three examples of well-formed save names for this moment in the game?"
    ),
    "expected_behavior": (
        "Must specify POST /save with a descriptive name string. "
        "Save name format: lowercase words joined by underscores, "
        "describing the current location or event. "
        "Good examples for this moment: 'vermilion_arrival', 'before_ss_anne', "
        "'ss_anne_entrance'. "
        "Must distinguish this from the existing examples in the skill: "
        "before_brock, route1_start, mt_moon_entrance, got_cut. "
        "Must also note: saves should happen every 15–20 turns during regular play "
        "AND always before gym battles, rival encounters, risky fights, "
        "entering a new town or dungeon, or any uncertain action. "
        "Must use the exact endpoint POST /save (not '/save_game', '/checkpoint', etc.)."
    ),
    "difficulty": "hard",
    "category": "api-procedure",
    "source": "golden",
}

example_05 = {
    "task_input": (
        "You are a Pokemon-playing agent. The user wants to watch you play in their browser. "
        "Describe the COMPLETE tunnel setup procedure: "
        "the exact SSH command structure, what host/user to connect to, "
        "which ports to forward, how to find the URL, and what to append to it."
    ),
    "expected_behavior": (
        "Must describe: use SSH to connect to nokey@localhost.run, "
        "forwarding local port 9876 to remote port 80. "
        "Redirect SSH output to a log file, wait 10 seconds for the tunnel to establish. "
        "Then grep the log for a URL ending in .lhr.life — "
        "this is the public URL assigned by localhost.run. "
        "Append /dashboard/ to that URL and give it to the user. "
        "Must warn: the .lhr.life URL changes every time the tunnel is restarted, "
        "so always re-grep the log for a fresh URL after any restart. "
        "Must use exact values: nokey@localhost.run, port 9876, .lhr.life pattern, /dashboard/ suffix. "
        "A generic SSH tunneling description without these specific values is insufficient."
    ),
    "difficulty": "hard",
    "category": "api-procedure",
    "source": "golden",
}

example_06 = {
    "task_input": (
        "You are a Pokemon-playing agent. During a play session you learn several things: "
        "(A) You are stuck at a ledge at y=28 and must go right to bypass it. "
        "(B) Your Squirtle is level 18 with Tackle and Water Gun. "
        "(C) Your current objective is to reach Mt. Moon. "
        "(D) In Pewter City, the Pokemon Center is to the west of the gym. "
        "For each piece of information, state which PKM memory prefix to use "
        "and write the exact memory entry."
    ),
    "expected_behavior": (
        "Must use the correct prefix for each: "
        "(A) PKM:STUCK — stuck situation with fix: 'PKM:STUCK ledge at y=28 go right to bypass' "
        "(B) PKM:TEAM — team notes: 'PKM:TEAM Squirtle Lv18, Tackle + Water Gun' "
        "(C) PKM:OBJECTIVE — current goal: 'PKM:OBJECTIVE Reach Mt. Moon' "
        "(D) PKM:MAP — navigation knowledge: 'PKM:MAP Pewter City: Pokemon Center is west of gym' "
        "Must use the exact prefixes PKM:STUCK, PKM:TEAM, PKM:OBJECTIVE, PKM:MAP "
        "(not 'NOTE:', 'MEMORY:', 'LOG:', or free-form text). "
        "The prefix table has 6 entries total: PKM:OBJECTIVE, PKM:MAP, PKM:STRATEGY, "
        "PKM:PROGRESS, PKM:STUCK, PKM:TEAM."
    ),
    "difficulty": "hard",
    "category": "api-procedure",
    "source": "golden",
}

example_07 = {
    "task_input": (
        "You are a Pokemon-playing agent. You are navigating through Viridian Forest "
        "heading north toward Pewter City. "
        "How many walk_up actions should be in a single POST /action call? "
        "What is the maximum before you must stop and re-observe? "
        "What happens if you send 12 walk actions at once?"
    ),
    "expected_behavior": (
        "Must state: send 2–4 movement actions per POST /action call (not 10–15). "
        "Must take a screenshot and re-check state after every 2–4 step batch. "
        "Maximum without re-checking: 4–5 actions. "
        "What happens with 12 at once: the character will overshoot intended positions, "
        "potentially walk off ledges, enter wrong buildings, or get stuck against obstacles — "
        "all while the agent has no knowledge of what happened because it skipped observation. "
        "The agent will not know where it ended up. "
        "Must also explain the vision verification rule: after each batch, "
        "GET /screenshot → save to /tmp/pokemon.png → vision_analyze to confirm "
        "the character moved where intended. "
        "Without this the agent WILL get lost, per the skill's explicit warning."
    ),
    "difficulty": "hard",
    "category": "api-procedure",
    "source": "golden",
}

example_08 = {
    "task_input": (
        "You are setting up the Pokemon agent server for the first time on this machine. "
        "List: (1) the exact working directory to cd into, "
        "(2) the venv activation command, "
        "(3) the exact server start command with ROM path and port, "
        "(4) how long to wait before verifying, "
        "(5) the exact API call to verify the server is healthy."
    ),
    "expected_behavior": (
        "Must give exact values from the skill: "
        "(1) cd /home/teknium/pokemon-agent "
        "(2) source .venv/bin/activate "
        "(3) pokemon-agent serve --rom roms/pokemon_red.gb --port 9876 & "
        "    (the & runs it in the background) "
        "(4) wait 4 seconds for startup "
        "(5) GET /health — this endpoint confirms the server is running. "
        "Must use port 9876 specifically (not 8080, 5000, or any other default). "
        "Must use the ROM path roms/pokemon_red.gb (relative to the working directory). "
        "A response that gives generic instructions without these exact values "
        "(directory, port, ROM path, wait time, health endpoint) is insufficient."
    ),
    "difficulty": "hard",
    "category": "api-procedure",
    "source": "golden",
}

GOLDEN_EXAMPLES_HARD = [
    example_01, example_02, example_03, example_04,
    example_05, example_06, example_07, example_08,
]
