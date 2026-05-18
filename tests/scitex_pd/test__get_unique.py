#!/usr/bin/env python3
"""Tests for scitex_pd._get_unique.get_unique."""

import runpy

import pandas as pd
import pytest

from scitex_pd._get_unique import get_unique


class TestSingleUniqueValue:
    def test_returns_the_unique_string_value(self):
        # Arrange
        df = pd.DataFrame({"patient_id": ["P01", "P01", "P01"]})
        # Act
        result = get_unique(df, "patient_id")
        # Assert
        assert result == "P01"

    def test_works_with_numeric_int_column(self):
        # Arrange
        df = pd.DataFrame({"x": [42, 42, 42]})
        # Act
        result = get_unique(df, "x")
        # Assert
        assert result == 42


class TestMultipleUniqueValues:
    def test_returns_default_value_when_default_specified(self):
        # Arrange
        df = pd.DataFrame({"patient_id": ["P01", "P02"]})
        # Act
        result = get_unique(df, "patient_id", default="Unknown")
        # Assert
        assert result == "Unknown"

    def test_returns_none_when_no_default_supplied(self):
        # Arrange
        df = pd.DataFrame({"patient_id": ["P01", "P02"]})
        # Act
        result = get_unique(df, "patient_id")
        # Assert
        assert result is None

    def test_raises_valueerror_when_raise_on_multiple_true(self):
        # Arrange
        df = pd.DataFrame({"patient_id": ["P01", "P02"]})
        # Act
        ctx = pytest.raises(ValueError, match="has 2 unique values")
        # Assert
        with ctx:
            get_unique(df, "patient_id", raise_on_multiple=True)

    def test_valueerror_message_includes_unique_count(self):
        # Arrange
        df = pd.DataFrame({"x": list("abcde")})
        try:
            get_unique(df, "x", raise_on_multiple=True)
            captured = None
        except ValueError as exc:
            captured = str(exc)
        # Act
        msg = captured
        # Assert
        assert msg is not None and "5 unique values" in msg


class TestMissingColumn:
    def test_returns_default_value_when_column_missing(self):
        # Arrange
        df = pd.DataFrame({"a": [1, 2, 3]})
        # Act
        result = get_unique(df, "missing", default="N/A")
        # Assert
        assert result == "N/A"

    def test_returns_none_when_missing_and_no_default(self):
        # Arrange
        df = pd.DataFrame({"a": [1, 2, 3]})
        # Act
        result = get_unique(df, "missing")
        # Assert
        assert result is None

    def test_raises_keyerror_when_missing_and_raise_flag_set(self):
        # Arrange
        df = pd.DataFrame({"a": [1, 2, 3]})
        # Act
        ctx = pytest.raises(KeyError, match="not found")
        # Assert
        with ctx:
            get_unique(df, "missing", raise_on_multiple=True)


class TestGetUniqueMainBlock:
    """Run the module-level `__main__` self-check via runpy so the demo
    block contributes real coverage instead of dead-on-import code."""

    def test_main_block_prints_all_tests_passed_marker(self, capsys):
        # Arrange
        module = "scitex_pd._get_unique"
        # Act
        runpy.run_module(module, run_name="__main__")
        captured = capsys.readouterr()
        # Assert
        assert "All tests passed" in captured.out


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])

# EOF
