# GOLDEN DATASET — 20 hand-crafted examples: 4 easy · 8 medium · 8 hard
#
# Scenario: pokemon-player  —  Pokemon Red/Blue/Yellow gameplay decisions
#
# Baseline skill: the current pokemon-player skill (full operational content)
#
# WHY this is a good evolution target
# ─────────────────────────────────────
# Easy examples (4): navigation/battle basics — skill covers these explicitly.
# Medium examples (8): general gameplay decisions — LLM partially knows.
# Hard examples (8): OPERATIONAL PROCEDURE questions requiring specific values
#   documented only in the skill.  The LLM cannot guess these from training:
#   - Exact API endpoints (GET /state, GET /screenshot, POST /action, POST /save,
#     GET /health) — LLM invents plausible-sounding but wrong paths
#   - Exact action names (wait_60, hold_b_120, a_until_dialog_end) — LLM guesses
#     variants that don't exist (wait, hold_B, dialog_skip, etc.)
#   - Server port 9876, working directory, ROM path — not guessable
#   - Screenshot save path /tmp/pokemon.png — not guessable
#   - PKM:* memory prefix table (PKM:STUCK, PKM:MAP, PKM:TEAM, …) — custom, unknown
#   - Tunnel host nokey@localhost.run and .lhr.life URL pattern — custom
#   - Action batch size rule (2–4 per call, max 4–5) — custom guidance
#
# Expected baseline holdout score:  ~0.10–0.25
# Expected evolved holdout score:   ~0.70–0.90

from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.scenarios.pokemon_player.golden_examples.easy import (
    GOLDEN_EXAMPLES_EASY,
)
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.scenarios.pokemon_player.golden_examples.medium import (
    GOLDEN_EXAMPLES_MEDIUM,
)
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.scenarios.pokemon_player.golden_examples.hard import (
    GOLDEN_EXAMPLES_HARD,
)

GOLDEN_EXAMPLES = GOLDEN_EXAMPLES_EASY + GOLDEN_EXAMPLES_MEDIUM + GOLDEN_EXAMPLES_HARD
