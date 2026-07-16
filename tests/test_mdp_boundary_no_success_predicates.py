"""Static guards against generic success/predicates MDP modules."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "ioailab"


def _py_files(*roots: Path):
    for root in roots:
        yield from (p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def test_no_success_or_predicates_modules_remain_anywhere():
    offenders = [
        p.relative_to(ROOT).as_posix()
        for name in ("success.py", "predicates.py")
        for p in SRC.rglob(name)
        if "__pycache__" not in p.parts
    ]
    assert offenders == []


def test_no_stale_mdp_success_or_predicates_imports():
    stale_tokens = (
        ".mdp." + "success",
        ".mdp." + "predicates",
        "mdp import " + "success",
        "mdp import " + "predicates",
    )
    this_file = Path(__file__).resolve()
    offenders: list[str] = []
    for path in _py_files(SRC, ROOT / "tests", ROOT / "examples"):
        if path.resolve() == this_file:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(token in text for token in stale_tokens):
            offenders.append(path.relative_to(ROOT).as_posix())
    assert offenders == []
