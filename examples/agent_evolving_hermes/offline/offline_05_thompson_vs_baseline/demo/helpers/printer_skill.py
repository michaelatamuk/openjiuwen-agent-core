from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.printer_banner import _banner


def _print_skill(label: str, text: str, max_lines: int = 35) -> None:
    _banner(label)
    lines = text.strip().splitlines()
    for line in lines[:max_lines]:
        print(line)
    if len(lines) > max_lines:
        print(f"  … ({len(lines) - max_lines} more lines not shown)")
