from __future__ import annotations

from pathlib import Path


def walk(file: Path) -> Path | None:
    for parent in [file, *file.parents]:
        if parent.is_dir():
            path = parent.joinpath('venv')

            if path.exists() and path.is_dir():
                return path.parent

    return None


file = Path.cwd()
CWD = walk(file)

ANIMATION = CWD.joinpath('animation')
ANIMATION.mkdir(exist_ok=True, parents=True)

DATA = CWD.joinpath('data')
DATA.mkdir(exist_ok=True, parents=True)

MODEL = CWD.joinpath('model')
MODEL.mkdir(exist_ok=True, parents=True)
