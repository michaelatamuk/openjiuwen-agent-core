from ._gate_maker import make_acceptance_gate


def apply_acceptance_gates(constraints_passed, config, min_improvement, improvement, evolved_score, skill_name,
                           evolved_text, cross_run_delta, output_dir, console):
    console.print("[blue]~~~ Evolving Stage 07 - Apply Acceptance Gates Started ~~~[/blue]")
    if not constraints_passed:
        accepted = False
        ts_conf = None
    else:
        gate = make_acceptance_gate(config, min_improvement)

        accepted, ts_conf = gate.decide(improvement, evolved_score, skill_name, evolved_text, cross_run_delta,
                                        output_dir, console)

    console.print("[blue]~~~ Evolving Stage 07 - Apply Acceptance Gates Finished ~~~[/blue]")
    return accepted, ts_conf
