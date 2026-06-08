from pathlib import Path


def _write_skill(skills_root: Path, name: str, frontmatter: str, body: str) -> Path:
    skill_dir = skills_root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_path = skill_dir / "SKILL.md"
    skill_path.write_text(f"---\n{frontmatter.strip()}\n---\n{body}", encoding="utf-8",)
    return skill_path
