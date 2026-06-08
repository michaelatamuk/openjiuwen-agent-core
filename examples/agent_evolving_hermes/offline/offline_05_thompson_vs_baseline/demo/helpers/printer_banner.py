def _banner(title: str, width: int = 100, run_index: int = 0, n_runs: int = 1, console=None) -> None:
    suffix = f" | run {run_index} of {n_runs}" if n_runs > 1 else ""
    console.print("\n" + "═" * width)
    console.print(f" {title}{suffix}")
    console.print("═" * width)
