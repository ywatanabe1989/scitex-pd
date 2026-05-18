#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for scitex_pd.force_df."""

import numpy as np
import pandas as pd
import pytest

from _helpers import frames_match
from scitex_pd import force_df


class TestForceDfBasic:
    """Test basic functionality of force_df."""

    def test_dict_to_dataframe_returns_dataframe_instance(self):
        # Arrange
        data = {"a": [1, 2, 3], "b": [4, 5, 6]}
        # Act
        result = force_df(data)
        # Assert
        assert isinstance(result, pd.DataFrame)

    def test_dict_to_dataframe_has_expected_shape(self):
        # Arrange
        data = {"a": [1, 2, 3], "b": [4, 5, 6]}
        # Act
        result = force_df(data)
        # Assert
        assert result.shape == (3, 2)

    def test_dict_to_dataframe_preserves_column_a_values(self):
        # Arrange
        data = {"a": [1, 2, 3], "b": [4, 5, 6]}
        # Act
        result = force_df(data)
        # Assert
        assert result["a"].tolist() == [1, 2, 3]

    def test_dataframe_passthrough_returns_equivalent_frame(self):
        # Arrange
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        # Act
        result = force_df(df)
        # Assert
        assert frames_match(result, df)

    def test_series_to_dataframe_keeps_series_name_as_column(self):
        # Arrange
        series = pd.Series([1, 2, 3], name="data")
        # Act
        result = force_df(series)
        # Assert
        assert result.columns[0] == "data"

    def test_series_to_dataframe_preserves_values(self):
        # Arrange
        series = pd.Series([1, 2, 3], name="data")
        # Act
        result = force_df(series)
        # Assert
        assert result["data"].tolist() == [1, 2, 3]

    def test_list_to_dataframe_uses_value_column_name(self):
        # Arrange
        data = [1, 2, 3, 4, 5]
        # Act
        result = force_df(data)
        # Assert
        assert result.columns[0] == "value"

    def test_list_to_dataframe_preserves_values_in_order(self):
        # Arrange
        data = [1, 2, 3, 4, 5]
        # Act
        result = force_df(data)
        # Assert
        assert result["value"].tolist() == [1, 2, 3, 4, 5]

    def test_tuple_to_dataframe_preserves_values_in_order(self):
        # Arrange
        data = (10, 20, 30)
        # Act
        result = force_df(data)
        # Assert
        assert result["value"].tolist() == [10, 20, 30]


class TestForceDfNumPy:
    """Test force_df with numpy arrays."""

    def test_1d_array_uses_value_column_and_preserves_values(self):
        # Arrange
        arr = np.array([1, 2, 3, 4])
        # Act
        result = force_df(arr)
        # Assert
        assert result["value"].tolist() == [1, 2, 3, 4]

    def test_2d_array_returns_two_by_three_dataframe(self):
        # Arrange
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        # Act
        result = force_df(arr)
        # Assert
        assert result.shape == (2, 3)

    def test_2d_array_uses_default_integer_column_names(self):
        # Arrange
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        # Act
        result = force_df(arr)
        # Assert
        assert list(result.columns) == [0, 1, 2]

    def test_empty_array_returns_zero_by_one_dataframe(self):
        # Arrange
        arr = np.array([])
        # Act
        result = force_df(arr)
        # Assert
        assert result.shape == (0, 1)


class TestForceDfMixedLengths:
    """Test force_df with mixed-length data."""

    def test_mixed_lengths_default_filler_pads_with_nan(self):
        # Arrange
        data = {"a": [1, 2, 3], "b": [4, 5], "c": [6]}
        # Act
        result = force_df(data)
        # Assert
        assert pd.isna(result["b"].iloc[2])

    def test_mixed_lengths_custom_zero_filler_pads_short_lists(self):
        # Arrange
        data = {"a": [1, 2, 3], "b": [4, 5], "c": [6]}
        # Act
        result = force_df(data, filler=0)
        # Assert
        assert result["b"].tolist() == [4, 5, 0]

    def test_scalar_values_in_dict_pad_with_nan_to_max_length(self):
        # Arrange
        data = {"a": 1, "b": [2, 3, 4], "c": "hello"}
        # Act
        result = force_df(data)
        # Assert
        assert pd.isna(result["a"].iloc[1])


class TestForceDfListedSeries:
    """Test force_df with list of Series."""

    def test_list_of_series_returns_three_by_one_nan_frame(self):
        # Arrange
        series1 = pd.Series({"a": 1, "b": 2})
        series2 = pd.Series({"a": 3, "b": 4})
        series3 = pd.Series({"a": 5, "b": 6})
        # Act
        result = force_df([series1, series2, series3])
        # Assert
        assert result["value"].isna().all()

    def test_dict_of_series_dict_creates_row_named_columns(self):
        # Arrange
        series1 = pd.Series({"a": 1, "b": 2})
        series2 = pd.Series({"a": 3, "b": 4})
        data = {"row_0": series1.to_dict(), "row_1": series2.to_dict()}
        # Act
        result = force_df(data)
        # Assert
        assert {"row_0", "row_1"}.issubset(result.columns)


class TestForceDfEdgeCases:
    """Test edge cases for force_df."""

    def test_empty_dict_returns_empty_dataframe(self):
        # Arrange
        data = {}
        # Act
        result = force_df(data)
        # Assert
        assert result.shape == (0, 0)

    def test_nested_lists_are_kept_as_object_values(self):
        # Arrange
        data = {"a": [1, 2], "b": [[3, 4], [5, 6]]}
        # Act
        result = force_df(data)
        # Assert
        assert result["b"].iloc[0] == [3, 4]

    def test_mixed_types_int_column_uses_int64_dtype(self):
        # Arrange
        data = {
            "int": [1, 2, 3],
            "float": [1.1, 2.2, 3.3],
            "str": ["a", "b", "c"],
            "bool": [True, False, True],
            "none": [None, None, None],
        }
        # Act
        result = force_df(data)
        # Assert
        assert result["int"].dtype == "int64"

    def test_mixed_types_bool_column_uses_bool_dtype(self):
        # Arrange
        data = {
            "int": [1, 2, 3],
            "bool": [True, False, True],
        }
        # Act
        result = force_df(data)
        # Assert
        assert result["bool"].dtype == "bool"

    def test_single_value_dict_returns_one_by_one_dataframe(self):
        # Arrange
        data = {"a": 42}
        # Act
        result = force_df(data)
        # Assert
        assert result["a"].iloc[0] == 42


class TestForceDfSpecialCases:
    """Test special cases and behaviors."""

    def test_series_without_name_uses_default_integer_column_name(self):
        # Arrange
        series = pd.Series([1, 2, 3])
        # Act
        result = force_df(series)
        # Assert
        assert result.columns[0] == 0

    def test_dict_with_none_scalar_value_pads_with_nan(self):
        # Arrange
        data = {"a": None, "b": [1, 2, 3]}
        # Act
        result = force_df(data)
        # Assert
        assert pd.isna(result["a"].iloc[0])

    def test_dict_with_unusual_string_keys_preserves_column_names(self):
        # Arrange
        data = {
            "column_1": [1, 2],
            "Column 2": [3, 4],
            "3rdColumn": [5, 6],
            "col-4": [7, 8],
        }
        # Act
        result = force_df(data)
        # Assert
        assert set(result.columns) == set(data.keys())

    def test_custom_string_filler_fills_short_columns_with_marker(self):
        # Arrange
        data = {"a": [1], "b": [2, 3]}
        # Act
        result = force_df(data, filler="missing")
        # Assert
        assert result["a"].iloc[1] == "missing"

    def test_custom_none_filler_pads_numeric_column_with_nan(self):
        # Arrange
        data = {"a": [1], "b": [2, 3]}
        # Act
        result = force_df(data, filler=None)
        # Assert
        assert pd.isna(result["a"].iloc[1])

    def test_custom_object_filler_pads_with_object_reference(self):
        # Arrange
        custom = object()
        data = {"a": [1], "b": [2, 3]}
        # Act
        result = force_df(data, filler=custom)
        # Assert
        assert result["a"].iloc[1] is custom


class TestForceDfIntegration:
    """Integration tests for force_df."""

    def test_real_world_mixed_length_dict_pads_with_string_filler(self):
        # Arrange
        data = {
            "experiment_id": [1, 2, 3],
            "measurements": [10.5, 20.3],
            "status": "completed",
            "notes": ["good", "better", "best", "excellent"],
        }
        # Act
        result = force_df(data, filler="N/A")
        # Assert
        assert result["measurements"].tolist() == [10.5, 20.3, "N/A", "N/A"]

    def test_force_df_then_fillna_supports_arithmetic_on_columns(self):
        # Arrange
        data = {"a": [1, 2], "b": [3, 4, 5]}
        # Act
        result = force_df(data).fillna(0)
        result["sum"] = result["a"] + result["b"]
        # Assert
        assert result["sum"].tolist() == [4, 6, 5]


class TestForceDfBranchCoverage:
    """Branches not exercised by the main behaviour suite above."""

    def test_none_input_returns_empty_dataframe(self):
        # Arrange
        data = None
        # Act
        result = force_df(data)
        # Assert
        assert result.empty

    def test_3d_ndarray_is_reshaped_to_two_dimensional_frame(self):
        # Arrange
        arr = np.arange(24).reshape(2, 3, 4)
        # Act
        result = force_df(arr)
        # Assert
        assert result.shape == (2, 12)

    def test_bool_scalar_input_returns_one_by_one_dataframe(self):
        # Arrange
        data = True
        # Act
        result = force_df(data)
        # Assert
        assert result.shape == (1, 1)

    def test_str_scalar_input_returns_one_row_with_string_value(self):
        # Arrange
        data = "hello"
        # Act
        result = force_df(data)
        # Assert
        assert result["value"].iloc[0] == "hello"

    def test_list_of_lists_input_returns_two_by_two_dataframe(self):
        # Arrange
        data = [[1, 2], [3, 4]]
        # Act
        result = force_df(data)
        # Assert
        assert result.shape == (2, 2)

    def test_set_input_falls_through_to_generic_iterable(self):
        # Arrange
        data = {1, 2, 3}
        # Act
        result = force_df(data)
        # Assert
        assert set(result["value"].tolist()) == {1, 2, 3}

    def test_unconvertible_object_raises_type_error_with_message(self):
        # Arrange
        class NotIterable:
            pass

        # Act
        ctx = pytest.raises(TypeError, match="Cannot convert object")
        # Assert
        with ctx:
            force_df(NotIterable())


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__)])
