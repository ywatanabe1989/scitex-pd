"""Smoke test: every example script under examples/ runs to completion."""

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = list(Path(__file__).parent.parent.joinpath("examples").glob("*.py"))


def test_examples_directory_contains_python_scripts():
    # Arrange
    examples = EXAMPLES
    # Act
    count = len(examples)
    # Assert
    assert count > 0, "No example scripts found under examples/"


@pytest.mark.parametrize("example", EXAMPLES, ids=[e.name for e in EXAMPLES])
def test_example_script_runs_to_completion_with_zero_exit(example, tmp_path):
    # Arrange
    cmd = [sys.executable, str(example)]
    # Act
    result = subprocess.run(
        cmd,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    # Assert
    assert result.returncode == 0, (
        f"{example.name} failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
