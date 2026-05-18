#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for scitex_pd.slice."""

import builtins

import numpy as np
import pandas as pd
import pytest

from _helpers import frames_match
from scitex_pd import slice as pd_slice


class TestSliceBasic:
    """Test basic functionality of slice function."""

    def test_slice_by_indices_returns_three_rows(self):
        # Arrange
        df = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": ["a", "b", "c", "d", "e"]})
        # Act
        result = pd_slice(df, builtins.slice(1, 4))
        # Assert
        assert len(result) == 3

    def test_slice_by_indices_returns_expected_column_a(self):
        # Arrange
        df = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": ["a", "b", "c", "d", "e"]})
        # Act
        result = pd_slice(df, builtins.slice(1, 4))
        # Assert
        assert result["A"].tolist() == [2, 3, 4]

    def test_slice_by_indices_returns_expected_column_b(self):
        # Arrange
        df = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": ["a", "b", "c", "d", "e"]})
        # Act
        result = pd_slice(df, builtins.slice(1, 4))
        # Assert
        assert result["B"].tolist() == ["b", "c", "d"]

    def test_slice_by_indices_preserves_original_index_positions(self):
        # Arrange
        df = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": ["a", "b", "c", "d", "e"]})
        # Act
        result = pd_slice(df, builtins.slice(1, 4))
        # Assert
        assert result.index.tolist() == [1, 2, 3]

    def test_slice_from_start_returns_two_rows(self):
        # Arrange
        df = pd.DataFrame({"A": [10, 20, 30, 40], "B": [100, 200, 300, 400]})
        # Act
        result = pd_slice(df, builtins.slice(None, 2))
        # Assert
        assert len(result) == 2

    def test_slice_from_start_returns_first_values(self):
        # Arrange
        df = pd.DataFrame({"A": [10, 20, 30, 40], "B": [100, 200, 300, 400]})
        # Act
        result = pd_slice(df, builtins.slice(None, 2))
        # Assert
        assert result["A"].tolist() == [10, 20]

    def test_slice_to_end_returns_two_rows(self):
        # Arrange
        df = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [10, 20, 30, 40, 50]})
        # Act
        result = pd_slice(df, builtins.slice(3, None))
        # Assert
        assert len(result) == 2

    def test_slice_to_end_returns_tail_values(self):
        # Arrange
        df = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [10, 20, 30, 40, 50]})
        # Act
        result = pd_slice(df, builtins.slice(3, None))
        # Assert
        assert result["A"].tolist() == [4, 5]

    def test_slice_with_step_two_returns_every_other_row(self):
        # Arrange
        df = pd.DataFrame({"A": list(range(10)), "B": list(range(10, 20))})
        # Act
        result = pd_slice(df, builtins.slice(0, 10, 2))
        # Assert
        assert result["A"].tolist() == [0, 2, 4, 6, 8]


class TestSliceByConditions:
    """Test slicing by conditions using dictionary."""

    def test_single_condition_returns_two_matching_rows(self):
        # Arrange
        df = pd.DataFrame(
            {"A": [1, 2, 3, 2, 1], "B": ["x", "y", "z", "y", "x"]}
        )
        # Act
        result = pd_slice(df, {"A": 2})
        # Assert
        assert result["A"].tolist() == [2, 2]

    def test_single_condition_preserves_companion_column_values(self):
        # Arrange
        df = pd.DataFrame(
            {"A": [1, 2, 3, 2, 1], "B": ["x", "y", "z", "y", "x"]}
        )
        # Act
        result = pd_slice(df, {"A": 2})
        # Assert
        assert result["B"].tolist() == ["y", "y"]

    def test_single_condition_preserves_source_index(self):
        # Arrange
        df = pd.DataFrame(
            {"A": [1, 2, 3, 2, 1], "B": ["x", "y", "z", "y", "x"]}
        )
        # Act
        result = pd_slice(df, {"A": 2})
        # Assert
        assert result.index.tolist() == [1, 3]

    def test_multiple_conditions_return_single_matching_row(self):
        # Arrange
        df = pd.DataFrame(
            {
                "A": [1, 1, 2, 2, 3],
                "B": ["x", "y", "x", "y", "x"],
                "C": [10, 20, 30, 40, 50],
            }
        )
        # Act
        result = pd_slice(df, {"A": 2, "B": "x"})
        # Assert
        assert result["C"].tolist() == [30]

    def test_list_condition_matches_each_listed_value(self):
        # Arrange
        df = pd.DataFrame(
            {"A": [1, 2, 3, 4, 5], "B": ["a", "b", "c", "d", "e"]}
        )
        # Act
        result = pd_slice(df, {"A": [2, 4, 5]})
        # Assert
        assert result["A"].tolist() == [2, 4, 5]


class TestColumnSlicing:
    """Test column selection functionality."""

    def test_select_single_column_drops_others(self):
        # Arrange
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
        # Act
        result = pd_slice(df, columns=["B"])
        # Assert
        assert list(result.columns) == ["B"]

    def test_select_single_column_preserves_values(self):
        # Arrange
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
        # Act
        result = pd_slice(df, columns=["B"])
        # Assert
        assert result["B"].tolist() == [4, 5, 6]

    def test_select_multiple_columns_returns_requested_subset(self):
        # Arrange
        df = pd.DataFrame(
            {"A": [1, 2], "B": [3, 4], "C": [5, 6], "D": [7, 8]}
        )
        # Act
        result = pd_slice(df, columns=["A", "C", "D"])
        # Assert
        assert list(result.columns) == ["A", "C", "D"]

    def test_reorder_columns_follows_supplied_order(self):
        # Arrange
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})
        # Act
        result = pd_slice(df, columns=["C", "A", "B"])
        # Assert
        assert list(result.columns) == ["C", "A", "B"]


class TestCombinedSlicing:
    """Test combining row and column slicing."""

    def test_slice_rows_and_columns_returns_subset_columns(self):
        # Arrange
        df = pd.DataFrame(
            {
                "A": [1, 2, 3, 4, 5],
                "B": [10, 20, 30, 40, 50],
                "C": ["a", "b", "c", "d", "e"],
                "D": [100, 200, 300, 400, 500],
            }
        )
        # Act
        result = pd_slice(df, builtins.slice(1, 4), columns=["B", "C"])
        # Assert
        assert list(result.columns) == ["B", "C"]

    def test_slice_rows_and_columns_returns_subset_rows_for_b(self):
        # Arrange
        df = pd.DataFrame(
            {
                "A": [1, 2, 3, 4, 5],
                "B": [10, 20, 30, 40, 50],
                "C": ["a", "b", "c", "d", "e"],
                "D": [100, 200, 300, 400, 500],
            }
        )
        # Act
        result = pd_slice(df, builtins.slice(1, 4), columns=["B", "C"])
        # Assert
        assert result["B"].tolist() == [20, 30, 40]

    def test_conditions_and_columns_filters_and_projects(self):
        # Arrange
        df = pd.DataFrame(
            {
                "category": ["A", "B", "A", "B", "A"],
                "value": [10, 20, 30, 40, 50],
                "extra1": [1, 2, 3, 4, 5],
                "extra2": [6, 7, 8, 9, 10],
            }
        )
        # Act
        result = pd_slice(df, {"category": "A"}, columns=["category", "value"])
        # Assert
        assert result["value"].tolist() == [10, 30, 50]


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dataframe_slice_returns_empty_dataframe(self):
        # Arrange
        df = pd.DataFrame()
        # Act
        result = pd_slice(df, builtins.slice(0, 10))
        # Assert
        assert result.empty

    def test_empty_dataframe_slice_returns_dataframe_type(self):
        # Arrange
        df = pd.DataFrame()
        # Act
        result = pd_slice(df, builtins.slice(0, 10))
        # Assert
        assert isinstance(result, pd.DataFrame)

    def test_no_conditions_returns_input_dataframe_unchanged(self):
        # Arrange
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        # Act
        result = pd_slice(df)
        # Assert
        assert frames_match(result, df)

    def test_no_matching_conditions_returns_zero_rows(self):
        # Arrange
        df = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})
        # Act
        result = pd_slice(df, {"A": 999})
        # Assert
        assert len(result) == 0

    def test_no_matching_conditions_preserves_column_schema(self):
        # Arrange
        df = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})
        # Act
        result = pd_slice(df, {"A": 999})
        # Assert
        assert list(result.columns) == ["A", "B"]

    def test_out_of_bounds_slice_returns_empty_dataframe(self):
        # Arrange
        df = pd.DataFrame({"A": [1, 2, 3]})
        # Act
        result = pd_slice(df, builtins.slice(10, 20))
        # Assert
        assert len(result) == 0

    def test_negative_out_of_bounds_slice_returns_empty_dataframe(self):
        # Arrange
        df = pd.DataFrame({"A": [1, 2, 3]})
        # Act
        result = pd_slice(df, builtins.slice(-10, -5))
        # Assert
        assert len(result) == 0

    def test_negative_slice_indices_return_two_rows(self):
        # Arrange
        df = pd.DataFrame(
            {"A": [1, 2, 3, 4, 5], "B": ["a", "b", "c", "d", "e"]}
        )
        # Act
        result = pd_slice(df, builtins.slice(-3, -1))
        # Assert
        assert result["A"].tolist() == [3, 4]


class TestDataTypes:
    """Test with various data types."""

    def test_mixed_dtypes_slice_preserves_int_values(self):
        # Arrange
        df = pd.DataFrame(
            {
                "int": [1, 2, 3, 4],
                "float": [1.1, 2.2, 3.3, 4.4],
                "str": ["a", "b", "c", "d"],
                "bool": [True, False, True, False],
                "date": pd.date_range("2023-01-01", periods=4),
            }
        )
        # Act
        result = pd_slice(df, builtins.slice(1, 3))
        # Assert
        assert result["int"].tolist() == [2, 3]

    def test_mixed_dtypes_slice_preserves_float_values(self):
        # Arrange
        df = pd.DataFrame(
            {
                "int": [1, 2, 3, 4],
                "float": [1.1, 2.2, 3.3, 4.4],
            }
        )
        # Act
        result = pd_slice(df, builtins.slice(1, 3))
        # Assert
        assert result["float"].tolist() == [2.2, 3.3]

    def test_mixed_dtypes_slice_preserves_bool_values(self):
        # Arrange
        df = pd.DataFrame(
            {
                "int": [1, 2, 3, 4],
                "bool": [True, False, True, False],
            }
        )
        # Act
        result = pd_slice(df, builtins.slice(1, 3))
        # Assert
        assert result["bool"].tolist() == [False, True]

    def test_nan_values_in_slice_are_preserved_at_correct_positions(self):
        # Arrange
        df = pd.DataFrame(
            {
                "A": [1, np.nan, 3, np.nan, 5],
                "B": ["a", "b", np.nan, "d", "e"],
            }
        )
        # Act
        result = pd_slice(df, builtins.slice(1, 4))
        # Assert
        assert pd.isna(result["A"].iloc[0]) and pd.isna(result["A"].iloc[2])


class TestIndexPreservation:
    """Test DataFrame index handling."""

    def test_custom_index_preservation_keeps_label_subset(self):
        # Arrange
        df = pd.DataFrame({"A": [1, 2, 3, 4]}, index=["w", "x", "y", "z"])
        # Act
        result = pd_slice(df, builtins.slice(1, 3))
        # Assert
        assert list(result.index) == ["x", "y"]

    def test_custom_index_preservation_supports_loc_lookup(self):
        # Arrange
        df = pd.DataFrame({"A": [1, 2, 3, 4]}, index=["w", "x", "y", "z"])
        # Act
        result = pd_slice(df, builtins.slice(1, 3))
        # Assert
        assert result.loc["x", "A"] == 2

    def test_multiindex_slice_returns_two_rows(self):
        # Arrange
        arrays = [["A", "A", "B", "B"], [1, 2, 1, 2]]
        index = pd.MultiIndex.from_arrays(arrays)
        df = pd.DataFrame({"value": [10, 20, 30, 40]}, index=index)
        # Act
        result = pd_slice(df, builtins.slice(1, 3))
        # Assert
        assert result["value"].tolist() == [20, 30]


class TestRealWorldScenarios:
    """Test real-world usage scenarios."""

    def test_data_filtering_workflow_returns_three_rows(self):
        # Arrange
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=10),
                "product": ["A", "B", "A", "C", "B", "A", "C", "B", "A", "C"],
                "quantity": [10, 20, 15, 5, 25, 30, 10, 35, 20, 15],
                "revenue": [100, 400, 150, 75, 500, 300, 150, 700, 200, 225],
            }
        )
        # Act
        result = pd_slice(df, {"product": "A"}, columns=["date", "product", "revenue"])
        filtered = result[result["revenue"] > 100]
        # Assert
        assert len(filtered) == 3

    def test_data_filtering_workflow_isolates_product_a(self):
        # Arrange
        df = pd.DataFrame(
            {
                "product": ["A", "B", "A", "C", "B", "A", "C", "B", "A", "C"],
                "revenue": [100, 400, 150, 75, 500, 300, 150, 700, 200, 225],
            }
        )
        # Act
        result = pd_slice(df, {"product": "A"}, columns=["product", "revenue"])
        # Assert
        assert (result["product"] == "A").all()

    def test_time_series_window_returns_twenty_four_rows(self):
        # Arrange
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=100, freq="h"),
                "value": np.random.randn(100),
            }
        )
        # Act
        result = pd_slice(df, builtins.slice(24, 48))
        # Assert
        assert len(result) == 24

    def test_time_series_window_starts_at_expected_hour_of_day(self):
        # Arrange
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=100, freq="h"),
                "value": np.random.randn(100),
            }
        )
        # Act
        result = pd_slice(df, builtins.slice(24, 48))
        # Assert
        assert result["timestamp"].iloc[0].day == 2


class TestDocstringExamples:
    """Test examples from the docstring."""

    def test_docstring_slice_example_returns_two_rows(self):
        # Arrange
        df = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "x"]})
        # Act
        result = pd_slice(df, builtins.slice(0, 2))
        # Assert
        assert result["A"].tolist() == [1, 2]

    def test_docstring_conditions_example_returns_single_row(self):
        # Arrange
        df = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "x"]})
        # Act
        result = pd_slice(df, {"A": [1, 2], "B": "x"})
        # Assert
        assert result["A"].tolist() == [1]

    def test_docstring_columns_example_returns_single_column(self):
        # Arrange
        df = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "x"]})
        # Act
        result = pd_slice(df, columns=["A"])
        # Assert
        assert list(result.columns) == ["A"]


class TestCopyBehavior:
    """Test that slice returns a copy, not a view."""

    def test_modifying_result_does_not_change_original(self):
        # Arrange
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        result = pd_slice(df, builtins.slice(0, 2))
        # Act
        result["A"] = [99, 98]
        # Assert
        assert df["A"].tolist() == [1, 2, 3]


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__)])
