import json

import random
from pathlib import Path


def _write_golden_dataset(skills_root: Path, name: str, examples: list) -> Path:
    """Write examples to the golden_dataset path that eval_source='golden' reads."""
    out = skills_root / name / "golden_dataset"
    out.mkdir(parents=True, exist_ok=True)

    shuffled = list(examples)
    random.shuffle(shuffled)
    n          = len(shuffled)
    train_end  = int(n * 0.50)
    val_end    = train_end + int(n * 0.25)
    splits     = {"train": shuffled[:train_end],
                  "val":   shuffled[train_end:val_end],
                  "holdout": shuffled[val_end:]}

    for split_name, rows in splits.items():
        path = out / f"{split_name}.jsonl"
        with open(path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    return out
