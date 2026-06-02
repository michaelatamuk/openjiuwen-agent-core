def _banner(title: str, width: int = 70, run_index: int = 0, n_runs: int = 1) -> None:
    suffix = f" | run {run_index} of {n_runs}" if n_runs > 1 else ""
    print("\n" + "═" * width)
    print(f" {title}{suffix}")
    print("═" * width)
