from pathlib import Path

import json


def _print_ts_insights(ts_state_dir: Path, skill_name: str, console) -> None:
    """Read TS arm state files and print a human-readable summary."""
    examples_path = ts_state_dir / f"ts_examples_{skill_name}.json"
    gate_path     = ts_state_dir / f"ts_gate_{skill_name}.json"

    if examples_path.exists():
        with open(examples_path) as fh:
            state = json.load(fh)
        arms = state.get("arms", {})
        if arms:
            console.print("\n  TS-TrainingSelector — Example Selector (arm state after training)")
            console.print("  Higher α/(α+β) = example was more discriminating, received more budget")
            ranked = sorted(
                [(k, v.get("alpha", 1.0), v.get("beta", 1.0)) for k, v in arms.items()],
                key=lambda x: x[1] / (x[1] + x[2]),
                reverse=True,
            )
            for rank, (key, alpha, beta) in enumerate(ranked, 1):
                ratio = alpha / (alpha + beta)
                bar   = "█" * int(ratio * 16) + "░" * (16 - int(ratio * 16))
                console.print(f"    #{rank}  {bar}  α={alpha:.1f} β={beta:.1f}  "
                              f"p={ratio:.2f}  key={key[:16]}…")

    if gate_path.exists():
        with open(gate_path) as fh:
            state = json.load(fh)
        cand = state.get("candidate", {})
        dep  = state.get("deployed",  {})
        if cand:
            ac, bc = cand.get("alpha", 1.0), cand.get("beta", 1.0)
            ad, bd = dep.get("alpha",  1.0), dep.get("beta", 1.0)
            console.print(f"\n  TS-AcceptanceGate")
            console.print(f"    Candidate arm : α={ac:.1f}, β={bc:.1f}  "
                          f"(mean={ac/(ac+bc):.2f})")
            console.print(f"    Deployed  arm : α={ad:.1f}, β={bd:.1f}  "
                          f"(mean={ad/(ad+bd):.2f})")
