#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for scitex_pd.to_numeric."""

import numpy as np
import pandas as pd
import pytest

from _helpers import series_match
from scitex_pd import to_numeric


@pytest.fixture
def mixed_df():
    """A DataFrame with mixed numeric/string columns used across the suite."""
    return pd.DataFrame(
        {
            "int_str": ["1", "2", "3", "4"],
            "float_str": ["1.5", "2.5", "3.5", "4.5"],
            "mixed": ["1", "2.5", "three", "4"],
            "pure_str": ["a", "b", "c", "d"],
            "already_int": [1, 2, 3, 4],
            "already_float": [1.1, 2.2, 3.3, 4.4],
            "with_nan": ["1", "2", np.nan, "4"],
        }
    )


@pytest.fixture
def datetime_df():
    """A DataFrame with datetime strings used by datetime-shape tests."""
    return pd.DataFrame(
        {
            "dates": ["2021-01-01", "2021-01-02", "2021-01-03"],
            "times": ["10:30:00", "11:45:00", "12:00:00"],
            "numbers": ["100", "200", "300"],
        }
    )


class TestToNumeric:
    """Test class for to_numeric function."""

    def test_to_numeric_symbol_is_callable_after_import(self):
        # Arrange
        from scitex_pd import to_numeric as imported
        # Act
        target = imported
        # Assert
        assert callable(target)

    def test_basic_conversion_converts_int_string_column_to_numeric(self, mixed_df):
        # Arrange
        df = mixed_df
        # Act
        result = to_numeric(df)
        # Assert
        assert pd.api.types.is_numeric_dtype(result["int_str"])

    def test_basic_conversion_preserves_int_string_values(self, mixed_df):
        # Arrange
        df = mixed_df
        # Act
        result = to_numeric(df)
        # Assert
        assert list(result["int_str"]) == [1, 2, 3, 4]

    def test_basic_conversion_converts_float_string_values(self, mixed_df):
        # Arrange
        df = mixed_df
        # Act
        result = to_numeric(df)
        # Assert
        assert list(result["float_str"]) == [1.5, 2.5, 3.5, 4.5]

    def test_coerce_mode_converts_invalid_mixed_value_to_nan(self, mixed_df):
        # Arrange
        df = mixed_df
        # Act
        result = to_numeric(df, errors="coerce")
        # Assert
        assert pd.isna(result["mixed"].iloc[2])

    def test_coerce_mode_converts_pure_string_column_to_all_nan(self, mixed_df):
        # Arrange
        df = mixed_df
        # Act
        result = to_numeric(df, errors="coerce")
        # Assert
        assert result["pure_str"].isna().all()

    def test_ignore_mode_keeps_pure_string_column_as_object_or_string(
        self, mixed_df
    ):
        # Arrange
        df = mixed_df
        # Act
        result = to_numeric(df, errors="ignore")
        # Assert
        assert list(result["pure_str"]) == ["a", "b", "c", "d"]

    def test_ignore_mode_keeps_mixed_column_as_object_or_string(self, mixed_df):
        # Arrange
        df = mixed_df
        # Act
        result = to_numeric(df, errors="ignore")
        # Assert
        assert list(result["mixed"]) == ["1", "2.5", "three", "4"]

    def test_raise_mode_raises_valueerror_for_invalid_column(self):
        # Arrange
        df = pd.DataFrame(
            {"valid": ["1", "2", "3"], "invalid": ["1", "two", "3"]}
        )
        # Act
        ctx = pytest.raises(ValueError)
        # Assert
        with ctx:
            to_numeric(df, errors="raise")

    def test_with_nan_in_column_preserves_nan_after_conversion(self, mixed_df):
        # Arrange
        df = mixed_df
        # Act
        result = to_numeric(df)
        # Assert
        assert pd.isna(result["with_nan"].iloc[2])

    def test_empty_dataframe_returns_empty_dataframe(self):
        # Arrange
        df = pd.DataFrame()
        # Act
        result = to_numeric(df)
        # Assert
        assert result.empty

    def test_empty_dataframe_with_columns_preserves_columns(self):
        # Arrange
        df = pd.DataFrame(columns=["A", "B"])
        # Act
        result = to_numeric(df)
        # Assert
        assert list(result.columns) == ["A", "B"]

    def test_single_column_dataframe_converts_to_numeric_dtype(self):
        # Arrange
        df = pd.DataFrame({"A": ["1", "2", "3"]})
        # Act
        result = to_numeric(df)
        # Assert
        assert pd.api.types.is_numeric_dtype(result["A"])

    def test_scientific_notation_string_converts_to_thousand(self):
        # Arrange
        df = pd.DataFrame(
            {
                "sci": ["1e3", "2.5e-2", "3E+4"],
                "normal": ["1000", "0.025", "30000"],
            }
        )
        # Act
        result = to_numeric(df)
        # Assert
        assert result["sci"].iloc[0] == 1000

    def test_boolean_word_strings_become_nan_under_coerce(self):
        # Arrange
        df = pd.DataFrame(
            {
                "bool_str": ["True", "False", "True"],
                "bool_num": ["1", "0", "1"],
            }
        )
        # Act
        result = to_numeric(df, errors="coerce")
        # Assert
        assert result["bool_str"].isna().all()

    def test_boolean_number_strings_convert_to_int(self):
        # Arrange
        df = pd.DataFrame(
            {
                "bool_str": ["True", "False", "True"],
                "bool_num": ["1", "0", "1"],
            }
        )
        # Act
        result = to_numeric(df, errors="coerce")
        # Assert
        assert list(result["bool_num"]) == [1, 0, 1]

    def test_already_numeric_columns_remain_numeric(self):
        # Arrange
        df = pd.DataFrame(
            {
                "int32": pd.array([1, 2, 3], dtype="int32"),
                "float32": pd.array([1.1, 2.2, 3.3], dtype="float32"),
                "int64": pd.array([1, 2, 3], dtype="int64"),
                "float64": pd.array([1.1, 2.2, 3.3], dtype="float64"),
            }
        )
        # Act
        result = to_numeric(df)
        non_numeric = [
            c for c in df.columns if not pd.api.types.is_numeric_dtype(result[c])
        ]
        # Assert
        assert non_numeric == []

    def test_whitespace_in_numeric_strings_is_stripped_by_pandas(self):
        # Arrange
        df = pd.DataFrame(
            {
                "with_spaces": ["  1  ", " 2.5 ", "3", "  4.0"],
                "with_tabs": ["\t1\t", "2\t", "\t3", "4\t\t"],
            }
        )
        # Act
        result = to_numeric(df)
        # Assert
        assert list(result["with_spaces"]) == [1, 2.5, 3, 4.0]

    def test_tab_padded_numeric_strings_convert_correctly(self):
        # Arrange
        df = pd.DataFrame({"with_tabs": ["\t1\t", "2\t", "\t3", "4\t\t"]})
        # Act
        result = to_numeric(df)
        # Assert
        assert list(result["with_tabs"]) == [1, 2, 3, 4]

    def test_currency_dollar_strings_become_nan_under_coerce(self):
        # Arrange
        df = pd.DataFrame(
            {
                "dollars": ["$100", "$200.50", "$300"],
                "pounds": ["£100", "£200.50", "£300"],
            }
        )
        # Act
        result = to_numeric(df, errors="coerce")
        # Assert
        assert result["dollars"].isna().all()

    def test_currency_pound_strings_become_nan_under_coerce(self):
        # Arrange
        df = pd.DataFrame(
            {
                "dollars": ["$100", "$200.50", "$300"],
                "pounds": ["£100", "£200.50", "£300"],
            }
        )
        # Act
        result = to_numeric(df, errors="coerce")
        # Assert
        assert result["pounds"].isna().all()

    def test_percentage_strings_become_nan_under_coerce(self):
        # Arrange
        df = pd.DataFrame(
            {
                "percent": ["10%", "20.5%", "30%"],
                "decimal": ["0.1", "0.205", "0.3"],
            }
        )
        # Act
        result = to_numeric(df, errors="coerce")
        # Assert
        assert result["percent"].isna().all()

    def test_decimal_strings_convert_to_floats_under_coerce(self):
        # Arrange
        df = pd.DataFrame(
            {
                "percent": ["10%", "20.5%", "30%"],
                "decimal": ["0.1", "0.205", "0.3"],
            }
        )
        # Act
        result = to_numeric(df, errors="coerce")
        # Assert
        assert list(result["decimal"]) == [0.1, 0.205, 0.3]

    def test_copy_behaviour_does_not_modify_original_int_str_column(
        self, mixed_df
    ):
        # Arrange
        original_values = mixed_df["int_str"].copy()
        # Act
        to_numeric(mixed_df)
        # Assert
        assert series_match(mixed_df["int_str"], original_values)

    @pytest.mark.parametrize("errors", ["coerce", "ignore"])
    def test_nums_column_converts_to_numeric_for_both_error_modes(
        self, errors
    ):
        # Arrange
        df = pd.DataFrame(
            {"nums": ["1", "2", "3"], "mixed": ["1", "a", "3"]}
        )
        # Act
        result = to_numeric(df, errors=errors)
        # Assert
        assert pd.api.types.is_numeric_dtype(result["nums"])


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__)])
