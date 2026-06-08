def _print_metrics(metrics: dict, console) -> None:
    bs = metrics.get("baseline_score", 0.0)
    es = metrics.get("evolved_score",  0.0)
    delta = es - bs
    sign  = "+" if delta >= 0 else ""
    console.print(f"    Pre-train score : {bs:.4f}")
    console.print(f"    Evolved score   : {es:.4f}")
    console.print(f"    Improvement     : {sign}{delta:.4f}")
    console.print(f"    Elapsed         : {metrics.get('elapsed_seconds', 0.0):.1f}s")
    console.print(f"    Accepted        : {'✓ Yes' if metrics.get('accepted') else '✗ No (saved as REGRESSION)'}")
    xd = metrics.get("cross_run_delta")
    if xd is not None:
        console.print(f"    vs prior best   : {'+' if xd>=0 else ''}{xd:.4f}")
    checks = metrics.get("constraint_checks", [])
    if checks:
        console.print("    Constraints:")
        for check in checks:
            icon = "✓" if check["passed"] else "✗"
            console.print(f"      {icon} {check['name']:20s}  {check['message']}")
