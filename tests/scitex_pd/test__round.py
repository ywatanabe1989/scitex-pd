#!/usr/bin/env python3
# Time-stamp: "2025-05-31 20:45:00 (ywatanabe)"
"""Comprehensive tests for scitex_pd.round."""

import numpy as np
import pandas as pd
import pytest

from _helpers import frames_match
from scitex_pd import round as pd_round


class TestRound:
    """Test class for round function."""

    def test_basic_float_rounding_returns_two_decimal_places(self):
        # Arrange
        df = pd.DataFrame(
            {"A": [1.23456, 2.34567, 3.45678], "B": [4.56789, 5.67890, 6.78901]}
        )
        expected = pd.DataFrame({"A": [1.23, 2.35, 3.46], "B": [4.57, 5.68, 6.79]})
        # Act
        result = pd_round(df, factor=2)
        # Assert
        assert frames_match(result, expected)

    def test_default_factor_rounds_to_three_decimals(self):
        # Arrange
        df = pd.DataFrame({"value": [1.234567, 2.345678, 3.456789]})
        expected = pd.DataFrame({"value": [1.235, 2.346, 3.457]})
        # Act
        result = pd_round(df)
        # Assert
        assert frames_match(result, expected)

    def test_mixed_types_rounds_only_numeric_columns(self):
        # Arrange
        df = pd.DataFrame(
            {
                "float": [1.23456, 2.34567],
                "int": [3, 4],
                "str": ["abc", "def"],
                "bool": [True, False],
            }
        )
        expected = pd.DataFrame(
            {
                "float": [1.23, 2.35],
                "int": [3, 4],
                "str": ["abc", "def"],
                # Booleans are converted to int by pd.to_numeric.
                "bool": [1, 0],
            }
        )
        # Act
        result = pd_round(df, factor=2)
        # Assert
        assert frames_match(result, expected)

    def test_integer_columns_preserve_int64_dtype(self):
        # Arrange
        df = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [10, 20, 30, 40, 50]})
        # Act
        result = pd_round(df, factor=2)
        # Assert
        assert (result["A"].dtype, result["B"].dtype) == (np.int64, np.int64)

    def test_integer_columns_round_returns_input_unchanged(self):
        # Arrange
        df = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [10, 20, 30, 40, 50]})
        # Act
        result = pd_round(df, factor=2)
        # Assert
        assert frames_match(result, df)

    def test_zero_decimal_places_uses_bankers_rounding(self):
        # Arrange
        df = pd.DataFrame({"A": [1.4, 2.5, 3.6], "B": [4.4, 5.5, 6.6]})
        expected = pd.DataFrame({"A": [1, 2, 4], "B": [4, 6, 7]})
        # Act
        result = pd_round(df, factor=0)
        # Assert
        assert frames_match(result, expected)

    def test_large_factor_six_keeps_six_decimal_digits(self):
        # Arrange
        df = pd.DataFrame({"A": [1.123456789, 2.234567890]})
        expected = pd.DataFrame({"A": [1.123457, 2.234568]})
        # Act
        result = pd_round(df, factor=6)
        # Assert
        assert frames_match(result, expected)

    def test_negative_values_round_to_same_precision(self):
        # Arrange
        df = pd.DataFrame(
            {"A": [-1.23456, -2.34567, -3.45678], "B": [1.23456, -2.34567, 3.45678]}
        )
        expected = pd.DataFrame(
            {"A": [-1.23, -2.35, -3.46], "B": [1.23, -2.35, 3.46]}
        )
        # Act
        result = pd_round(df, factor=2)
        # Assert
        assert frames_match(result, expected)

    def test_nan_values_are_preserved_while_finite_round(self):
        # Arrange
        df = pd.DataFrame(
            {
                "A": [1.234, np.nan, 3.456],
                "B": [np.nan, 2.345, np.nan],
                "C": [1.234, 2.345, 3.456],
            }
        )
        expected = pd.DataFrame(
            {
                "A": [1.23, np.nan, 3.46],
                "B": [np.nan, 2.35, np.nan],
                "C": [1.23, 2.35, 3.46],
            }
        )
        # Act
        result = pd_round(df, factor=2)
        # Assert
        assert frames_match(result, expected)

    def test_inf_values_are_preserved_while_finite_round(self):
        # Arrange
        df = pd.DataFrame(
            {
                "A": [1.234, np.inf, -np.inf],
                "B": [np.inf, 2.345, -np.inf],
                "C": [1.234, 2.345, 3.456],
            }
        )
        expected = pd.DataFrame(
            {
                "A": [1.23, np.inf, -np.inf],
                "B": [np.inf, 2.35, -np.inf],
                "C": [1.23, 2.35, 3.46],
            }
        )
        # Act
        result = pd_round(df, factor=2)
        # Assert
        assert frames_match(result, expected)

    def test_empty_dataframe_returns_empty_dataframe(self):
        # Arrange
        df = pd.DataFrame()
        # Act
        result = pd_round(df, factor=2)
        # Assert
        assert frames_match(result, df)

    def test_single_column_dataframe_rounds_to_factor(self):
        # Arrange
        df = pd.DataFrame({"values": [1.234567, 2.345678, 3.456789]})
        expected = pd.DataFrame({"values": [1.235, 2.346, 3.457]})
        # Act
        result = pd_round(df, factor=3)
        # Assert
        assert frames_match(result, expected)

    def test_datetime_columns_are_returned_unchanged(self):
        # Arrange
        dates = pd.date_range("2024-01-01", periods=3)
        df = pd.DataFrame({"date": dates, "value": [1.23456, 2.34567, 3.45678]})
        expected = pd.DataFrame({"date": dates, "value": [1.23, 2.35, 3.46]})
        # Act
        result = pd_round(df, factor=2)
        # Assert
        assert frames_match(result, expected)

    def test_categorical_columns_are_returned_unchanged(self):
        # Arrange
        df = pd.DataFrame(
            {
                "category": pd.Categorical(["A", "B", "C"]),
                "value": [1.23456, 2.34567, 3.45678],
            }
        )
        expected = pd.DataFrame(
            {
                "category": pd.Categorical(["A", "B", "C"]),
                "value": [1.23, 2.35, 3.46],
            }
        )
        # Act
        result = pd_round(df, factor=2)
        # Assert
        assert frames_match(result, expected)

    def test_scientific_notation_values_round_to_factor(self):
        # Arrange
        df = pd.DataFrame(
            {
                "A": [1.234e-5, 2.345e-5, 3.456e-5],
                "B": [1.234e5, 2.345e5, 3.456e5],
            }
        )
        expected = pd.DataFrame(
            {"A": [0.0, 0.0, 0.0], "B": [123400.0, 234500.0, 345600.0]}
        )
        # Act
        result = pd_round(df, factor=3)
        # Assert
        assert frames_match(result, expected)

    def test_very_small_values_round_to_zero_at_factor_three(self):
        # Arrange
        df = pd.DataFrame({"A": [0.000123456, 0.000234567, 0.000345678]})
        expected = pd.DataFrame({"A": [0.0, 0.0, 0.0]})
        # Act
        result = pd_round(df, factor=3)
        # Assert
        assert frames_match(result, expected)

    def test_factor_zero_converts_round_floats_to_int(self):
        # Arrange
        df = pd.DataFrame(
            {"A": [1.00001, 2.00002, 3.00003], "B": [4.99999, 5.99998, 6.99997]}
        )
        expected = pd.DataFrame({"A": [1, 2, 3], "B": [5, 6, 7]})
        # Act
        result = pd_round(df, factor=0)
        # Assert
        assert frames_match(result, expected)

    def test_multiindex_dataframe_round_preserves_index(self):
        # Arrange
        arrays = [["A", "A", "B", "B"], [1, 2, 1, 2]]
        index = pd.MultiIndex.from_arrays(arrays, names=["first", "second"])
        df = pd.DataFrame(
            {"value": [1.23456, 2.34567, 3.45678, 4.56789]}, index=index
        )
        expected = pd.DataFrame(
            {"value": [1.23, 2.35, 3.46, 4.57]}, index=index
        )
        # Act
        result = pd_round(df, factor=2)
        # Assert
        assert frames_match(result, expected)

    def test_object_string_column_is_returned_unchanged(self):
        # Arrange
        df = pd.DataFrame(
            {"A": ["1.234", "2.345", "3.456"], "B": [1.234, 2.345, 3.456]}
        )
        expected = pd.DataFrame(
            {"A": ["1.234", "2.345", "3.456"], "B": [1.23, 2.35, 3.46]}
        )
        # Act
        result = pd_round(df, factor=2)
        # Assert
        assert frames_match(result, expected)

    def test_none_values_round_to_nan_while_finite_round(self):
        # Arrange
        df = pd.DataFrame(
            {
                "A": [1.234, None, 3.456],
                "B": [None, 2.345, None],
                "C": [1.234, 2.345, 3.456],
            }
        )
        expected = pd.DataFrame(
            {
                "A": [1.23, np.nan, 3.46],
                "B": [np.nan, 2.35, np.nan],
                "C": [1.23, 2.35, 3.46],
            }
        )
        # Act
        result = pd_round(df, factor=2)
        # Assert
        assert frames_match(result, expected)

    def test_object_dtype_with_numbers_is_returned_unchanged(self):
        # Arrange
        df = pd.DataFrame(
            {
                "A": pd.Series([1.234, 2.345, 3.456], dtype="object"),
                "B": pd.Series(["a", "b", "c"], dtype="object"),
            }
        )
        expected = pd.DataFrame(
            {
                "A": pd.Series([1.234, 2.345, 3.456], dtype="object"),
                "B": pd.Series(["a", "b", "c"], dtype="object"),
            }
        )
        # Act
        result = pd_round(df, factor=2)
        # Assert
        assert frames_match(result, expected)

    def test_round_preserves_original_column_order(self):
        # Arrange
        df = pd.DataFrame({"Z": [1.234], "A": [2.345], "M": [3.456]})
        # Act
        result = pd_round(df, factor=2)
        # Assert
        assert list(result.columns) == ["Z", "A", "M"]

    def test_round_column_z_uses_supplied_factor(self):
        # Arrange
        df = pd.DataFrame({"Z": [1.234], "A": [2.345], "M": [3.456]})
        # Act
        result = pd_round(df, factor=2)
        # Assert
        assert result["Z"][0] == 1.23

    def test_round_column_a_uses_supplied_factor(self):
        # Arrange
        df = pd.DataFrame({"Z": [1.234], "A": [2.345], "M": [3.456]})
        # Act
        result = pd_round(df, factor=2)
        # Assert
        assert result["A"][0] == 2.35

    def test_round_column_m_uses_supplied_factor(self):
        # Arrange
        df = pd.DataFrame({"Z": [1.234], "A": [2.345], "M": [3.456]})
        # Act
        result = pd_round(df, factor=2)
        # Assert
        assert result["M"][0] == 3.46

    def test_round_preserves_dataframe_shape_on_large_input(self):
        # Arrange
        np.random.seed(42)
        df = pd.DataFrame(np.random.randn(1000, 10))
        # Act
        result = pd_round(df, factor=3)
        # Assert
        assert result.shape == df.shape

    def test_round_matches_np_round_on_large_input(self):
        # Arrange
        np.random.seed(42)
        df = pd.DataFrame(np.random.randn(1000, 10))
        # Act
        result = pd_round(df, factor=3)
        # Assert
        assert abs(result.iloc[0, 0] - np.round(df.iloc[0, 0], 3)) < 1e-10

    def test_factor_one_rounds_to_one_decimal_place(self):
        # Arrange
        df = pd.DataFrame({"A": [1.234, 2.567, 3.891]})
        expected = pd.DataFrame({"A": [1.2, 2.6, 3.9]})
        # Act
        result = pd_round(df, factor=1)
        # Assert
        assert frames_match(result, expected)

    def test_complex_mixed_dataframe_rounds_only_numeric(self):
        # Arrange
        df = pd.DataFrame(
            {
                "floats": [1.23456, 2.34567, np.nan],
                "floats_no_nan": [1.23456, 2.34567, 3.45678],
                "ints": [1, 2, 3],
                "strings": ["a", "b", "c"],
                "bools": [True, False, True],
                "mixed": [1.234, "text", None],
            }
        )
        expected = pd.DataFrame(
            {
                "floats": [1.23, 2.35, np.nan],
                "floats_no_nan": [1.23, 2.35, 3.46],
                "ints": [1, 2, 3],
                "strings": ["a", "b", "c"],
                "bools": [1, 0, 1],
                # Object dtype with mixed types — returned unchanged.
                "mixed": [1.234, "text", None],
            }
        )
        # Act
        result = pd_round(df, factor=2)
        # Assert
        assert frames_match(result, expected)

    def test_half_values_use_bankers_round_to_even(self):
        # Arrange
        df = pd.DataFrame({"A": [1.125, 2.225, 3.335, 4.445, 5.555]})
        expected = pd.DataFrame({"A": [1.12, 2.22, 3.34, 4.44, 5.56]})
        # Act
        result = pd_round(df, factor=2)
        # Assert
        assert frames_match(result, expected)

    def test_round_preserves_custom_string_index(self):
        # Arrange
        df = pd.DataFrame({"A": [1.234, 2.345, 3.456]}, index=["x", "y", "z"])
        # Act
        result = pd_round(df, factor=2)
        # Assert
        assert list(result.index) == ["x", "y", "z"]

    def test_round_uses_factor_per_column(self):
        # Arrange
        df = pd.DataFrame(
            {"precise": [1.123456789, 2.234567890], "rough": [100.1, 200.2]}
        )
        expected = pd.DataFrame(
            {"precise": [1.1235, 2.2346], "rough": [100.1, 200.2]}
        )
        # Act
        result = pd_round(df, factor=4)
        # Assert
        assert frames_match(result, expected)


class TestRoundFallback:
    """Defensive ValueError/TypeError fallback inside `custom_round`."""

    def test_nullable_bool_with_na_returns_column_unchanged(self):
        # Arrange
        # `astype(int)` on a nullable-boolean Series containing pd.NA
        # raises IntCastingNaNError (a ValueError subclass), exercising
        # the except branch.
        col = pd.Series([True, False, pd.NA], dtype="boolean")
        df = pd.DataFrame({"A": col})
        # Act
        out = pd_round(df)
        # Assert
        assert out["A"].dtype == col.dtype


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__)])
