from __future__ import annotations

import subprocess
from pathlib import Path


def run_command(command: list[str]) -> None:
    result = subprocess.run(command, check=False, capture_output=True, text=True)

    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        message = stderr or stdout or f"Command failed: {' '.join(command)}"
        raise RuntimeError(message)


def dvc_add(path: Path) -> None:
    run_command(["dvc", "add", str(path)])


def dvc_push() -> None:
    run_command(["dvc", "push"])