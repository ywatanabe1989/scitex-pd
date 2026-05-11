#!/usr/bin/env python3
"""Tests for scitex_pd._get_unique.get_unique."""

import pandas as pd
import pytest

from scitex_pd._get_unique import get_unique


class TestSingleUniqueValue:
    def test_returns_the_unique_value(self):
        df = pd.DataFrame({"patient_id": ["P01", "P01", "P01"]})
        assert get_unique(df, "patient_id") == "P01"

    def test_works_with_numeric_column(self):
        df = pd.DataFrame({"x": [42, 42, 42]})
        assert get_unique(df, "x") == 42


class TestMultipleUniqueValues:
    def test_returns_default_when_default_specified(self):
        df = pd.DataFrame({"patient_id": ["P01", "P02"]})
        assert get_unique(df, "patient_id", default="Unknown") == "Unknown"

    def test_returns_none_when_no_default(self):
        df = pd.DataFrame({"patient_id": ["P01", "P02"]})
        assert get_unique(df, "patient_id") is None

    def test_raises_when_raise_on_multiple_set(self):
        df = pd.DataFrame({"patient_id": ["P01", "P02"]})
        with pytest.raises(ValueError, match="has 2 unique values"):
            get_unique(df, "patient_id", raise_on_multiple=True)

    def test_value_error_message_includes_sample(self):
        df = pd.DataFrame({"x": list("abcde")})
        with pytest.raises(ValueError) as ex:
            get_unique(df, "x", raise_on_multiple=True)
        msg = str(ex.value)
        assert "5 unique values" in msg


class TestMissingColumn:
    def test_returns_default_when_default_specified(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        assert get_unique(df, "missing", default="N/A") == "N/A"

    def test_returns_none_when_no_default_and_no_raise(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        assert get_unique(df, "missing") is None

    def test_raises_keyerror_when_raise_on_multiple_and_missing(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(KeyError, match="not found"):
            get_unique(df, "missing", raise_on_multiple=True)



import runpy


class TestGetUniqueMainBlock:
    """Run the module-level `__main__` self-check via runpy so the demo
    block contributes real coverage instead of dead-on-import code."""

    def test_main_block(self, capsys):
        runpy.run_module("scitex_pd._get_unique", run_name="__main__")
        captured = capsys.readouterr()
        assert "All tests passed" in captured.out


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])

# EOF
