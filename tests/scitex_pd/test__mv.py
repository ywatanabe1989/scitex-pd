#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for scitex_pd.mv / mv_to_first / mv_to_last."""

import numpy as np
import pandas as pd
import pytest

from scitex_pd import mv, mv_to_first, mv_to_last


class TestMvBasicFunctionality:
    """Test basic functionality of mv function."""

    def test_move_column_to_index_two_yields_expected_order(self):
        # Arrange
        df = pd.DataFrame(
            {"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9], "D": [10, 11, 12]}
        )
        # Act
        result = mv(df, "B", 2)
        # Assert
        assert list(result.columns) == ["A", "C", "B", "D"]

    def test_move_column_to_index_two_preserves_column_values(self):
        # Arrange
        df = pd.DataFrame(
            {"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9], "D": [10, 11, 12]}
        )
        # Act
        result = mv(df, "B", 2)
        # Assert
        assert result["B"].tolist() == [4, 5, 6]

    def test_move_column_to_first_position_yields_expected_order(self):
        # Arrange
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})
        # Act
        result = mv(df, "C", 0)
        # Assert
        assert list(result.columns) == ["C", "A", "B"]

    def test_move_column_to_last_position_yields_expected_order(self):
        # Arrange
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})
        # Act
        result = mv(df, "A", -1)
        # Assert
        assert list(result.columns) == ["B", "C", "A"]

    def test_move_row_yields_expected_index_order(self):
        # Arrange
        df = pd.DataFrame(
            {"col1": [1, 2, 3, 4], "col2": [5, 6, 7, 8]},
            index=["a", "b", "c", "d"],
        )
        # Act
        result = mv(df, "c", 1, axis=0)
        # Assert
        assert list(result.index) == ["a", "c", "b", "d"]

    def test_move_row_preserves_data_for_moved_row(self):
        # Arrange
        df = pd.DataFrame(
            {"col1": [1, 2, 3, 4], "col2": [5, 6, 7, 8]},
            index=["a", "b", "c", "d"],
        )
        # Act
        result = mv(df, "c", 1, axis=0)
        # Assert
        assert result.loc["c", "col1"] == 3


class TestMvNegativePositions:
    """Test negative position handling."""

    def test_negative_one_position_moves_column_to_last(self):
        # Arrange
        df = pd.DataFrame({"A": [1], "B": [2], "C": [3], "D": [4], "E": [5]})
        # Act
        result = mv(df, "B", -1)
        # Assert
        assert list(result.columns) == ["A", "C", "D", "E", "B"]

    def test_negative_two_position_moves_column_to_second_last(self):
        # Arrange
        df = pd.DataFrame({"A": [1], "B": [2], "C": [3], "D": [4], "E": [5]})
        # Act
        result = mv(df, "B", -2)
        # Assert
        assert list(result.columns) == ["A", "C", "D", "B", "E"]

    def test_negative_three_position_moves_column_to_third_last(self):
        # Arrange
        df = pd.DataFrame({"A": [1], "B": [2], "C": [3], "D": [4], "E": [5]})
        # Act
        result = mv(df, "B", -3)
        # Assert
        assert list(result.columns) == ["A", "C", "B", "D", "E"]

    def test_negative_one_position_moves_row_to_last(self):
        # Arrange
        df = pd.DataFrame({"col": [1, 2, 3, 4]}, index=["a", "b", "c", "d"])
        # Act
        result = mv(df, "b", -1, axis=0)
        # Assert
        assert list(result.index) == ["a", "c", "d", "b"]

    def test_negative_two_position_moves_row_to_second_last(self):
        # Arrange
        df = pd.DataFrame({"col": [1, 2, 3, 4]}, index=["a", "b", "c", "d"])
        # Act
        result = mv(df, "b", -2, axis=0)
        # Assert
        assert list(result.index) == ["a", "c", "b", "d"]


class TestMvToFirst:
    """Test mv_to_first function."""

    def test_mv_to_first_column_yields_expected_order(self):
        # Arrange
        df = pd.DataFrame(
            {"A": [1, 2], "B": [3, 4], "C": [5, 6], "D": [7, 8]}
        )
        # Act
        result = mv_to_first(df, "C")
        # Assert
        assert list(result.columns) == ["C", "A", "B", "D"]

    def test_mv_to_first_when_already_first_keeps_order(self):
        # Arrange
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        # Act
        result = mv_to_first(df, "A")
        # Assert
        assert list(result.columns) == ["A", "B"]

    def test_mv_to_first_row_yields_expected_index_order(self):
        # Arrange
        df = pd.DataFrame({"col": [1, 2, 3, 4]}, index=["a", "b", "c", "d"])
        # Act
        result = mv_to_first(df, "c", axis=0)
        # Assert
        assert list(result.index) == ["c", "a", "b", "d"]


class TestMvToLast:
    """Test mv_to_last function."""

    def test_mv_to_last_column_yields_expected_order(self):
        # Arrange
        df = pd.DataFrame(
            {"A": [1, 2], "B": [3, 4], "C": [5, 6], "D": [7, 8]}
        )
        # Act
        result = mv_to_last(df, "B")
        # Assert
        assert list(result.columns) == ["A", "C", "D", "B"]

    def test_mv_to_last_when_already_last_keeps_order(self):
        # Arrange
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        # Act
        result = mv_to_last(df, "B")
        # Assert
        assert list(result.columns) == ["A", "B"]

    def test_mv_to_last_row_yields_expected_index_order(self):
        # Arrange
        df = pd.DataFrame({"col": [1, 2, 3, 4]}, index=["a", "b", "c", "d"])
        # Act
        result = mv_to_last(df, "b", axis=0)
        # Assert
        assert list(result.index) == ["a", "c", "d", "b"]


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_nonexistent_column_raises_valueerror(self):
        # Arrange
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        # Act
        ctx = pytest.raises(ValueError)
        # Assert
        with ctx:
            mv(df, "C", 0)

    def test_nonexistent_row_raises_valueerror(self):
        # Arrange
        df = pd.DataFrame({"A": [1, 2]}, index=["a", "b"])
        # Act
        ctx = pytest.raises(ValueError)
        # Assert
        with ctx:
            mv(df, "c", 0, axis=0)

    def test_single_column_to_position_zero_preserves_column(self):
        # Arrange
        df = pd.DataFrame({"A": [1, 2, 3]})
        # Act
        result = mv(df, "A", 0)
        # Assert
        assert list(result.columns) == ["A"]

    def test_single_column_to_negative_one_preserves_column(self):
        # Arrange
        df = pd.DataFrame({"A": [1, 2, 3]})
        # Act
        result = mv(df, "A", -1)
        # Assert
        assert list(result.columns) == ["A"]

    def test_empty_dataframe_move_raises_valueerror(self):
        # Arrange
        df = pd.DataFrame()
        # Act
        ctx = pytest.raises(ValueError)
        # Assert
        with ctx:
            mv(df, "A", 0)

    def test_position_beyond_end_places_column_at_end(self):
        # Arrange
        df = pd.DataFrame({"A": [1], "B": [2], "C": [3]})
        # Act
        result = mv(df, "A", 10)
        # Assert
        assert list(result.columns) == ["B", "C", "A"]

    def test_large_negative_position_places_column_at_beginning(self):
        # Arrange
        df = pd.DataFrame({"A": [1], "B": [2], "C": [3]})
        # Act
        result = mv(df, "C", -10)
        # Assert
        assert list(result.columns) == ["C", "A", "B"]


class TestDataTypes:
    """Test with different data types."""

    def test_mixed_column_types_move_preserves_column_order(self):
        # Arrange
        df = pd.DataFrame(
            {
                "int": [1, 2, 3],
                "float": [1.1, 2.2, 3.3],
                "str": ["a", "b", "c"],
                "bool": [True, False, True],
                "date": pd.date_range("2023-01-01", periods=3),
            }
        )
        # Act
        result = mv(df, "bool", 1)
        # Assert
        assert list(result.columns) == ["int", "bool", "float", "str", "date"]

    def test_mixed_column_types_move_preserves_data_values(self):
        # Arrange
        df = pd.DataFrame(
            {
                "int": [1, 2, 3],
                "float": [1.1, 2.2, 3.3],
                "str": ["a", "b", "c"],
                "bool": [True, False, True],
                "date": pd.date_range("2023-01-01", periods=3),
            }
        )
        # Act
        result = mv(df, "bool", 1)
        # Assert
        assert result["float"].tolist() == [1.1, 2.2, 3.3]

    def test_categorical_index_row_move_yields_expected_order(self):
        # Arrange
        df = pd.DataFrame(
            {"A": [1, 2, 3], "B": [4, 5, 6]},
            index=pd.Categorical(["x", "y", "z"]),
        )
        # Act
        result = mv(df, "y", 0, axis=0)
        # Assert
        assert list(result.index) == ["y", "x", "z"]

    def test_multiindex_columns_move_yields_target_column_first(self):
        # Arrange
        arrays = [["A", "A", "B", "B"], ["X", "Y", "X", "Y"]]
        columns = pd.MultiIndex.from_arrays(arrays)
        df = pd.DataFrame(np.random.randn(3, 4), columns=columns)
        # Act
        result = mv(df, ("B", "X"), 0)
        # Assert
        assert result.columns[0] == ("B", "X")


class TestIndexPreservation:
    """Test that indices and data are properly preserved."""

    def test_column_name_attribute_is_preserved_after_move(self):
        # Arrange
        df = pd.DataFrame(
            {
                "A": pd.Series([1, 2, 3], name="A"),
                "B": pd.Series([4, 5, 6], name="B"),
                "C": pd.Series([7, 8, 9], name="C"),
            }
        )
        # Act
        result = mv(df, "B", 0)
        # Assert
        assert result["B"].name == "B"

    def test_index_name_is_preserved_after_row_move(self):
        # Arrange
        df = pd.DataFrame(
            {"A": [1, 2, 3], "B": [4, 5, 6]},
            index=pd.Index(["x", "y", "z"], name="my_index"),
        )
        # Act
        result = mv(df, "y", 0, axis=0)
        # Assert
        assert result.index.name == "my_index"

    def test_original_dataframe_columns_unchanged_after_move(self):
        # Arrange
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
        original = list(df.columns)
        # Act
        mv(df, "B", 0)
        # Assert
        assert list(df.columns) == original


class TestComplexScenarios:
    """Test complex real-world scenarios."""

    def test_chained_mv_to_first_places_keys_in_expected_order(self):
        # Arrange
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["Alice", "Bob", "Charlie"],
                "age": [25, 30, 35],
                "score": [95, 87, 92],
                "category": ["A", "B", "A"],
            }
        )
        # Act
        result = mv_to_first(df, "category")
        result = mv_to_first(result, "id")
        # Assert
        assert list(result.columns) == ["id", "category", "name", "age", "score"]

    def test_three_sequential_moves_produce_expected_order(self):
        # Arrange
        df = pd.DataFrame(
            {"A": [1], "B": [2], "C": [3], "D": [4], "E": [5]}
        )
        # Act
        result = mv(df, "E", 0)
        result = mv(result, "C", 2)
        result = mv(result, "A", -1)
        # Assert
        assert list(result.columns) == ["E", "C", "B", "D", "A"]

    def test_pivot_style_reorganization_orders_grouping_columns_first(self):
        # Arrange
        df = pd.DataFrame(
            {
                "value1": [10, 20, 30],
                "value2": [40, 50, 60],
                "group": ["A", "B", "C"],
                "subgroup": ["X", "Y", "Z"],
                "metric1": [1.1, 2.2, 3.3],
                "metric2": [4.4, 5.5, 6.6],
            }
        )
        # Act
        result = mv_to_first(df, "subgroup")
        result = mv_to_first(result, "group")
        # Assert
        assert list(result.columns) == [
            "group",
            "subgroup",
            "value1",
            "value2",
            "metric1",
            "metric2",
        ]


class TestNaNAndSpecialValues:
    """Test handling of NaN and special values."""

    def test_nan_in_moved_column_remains_in_position_zero(self):
        # Arrange
        df = pd.DataFrame(
            {"A": [1, np.nan, 3], "B": [np.nan, 5, 6], "C": [7, 8, np.nan]}
        )
        # Act
        result = mv(df, "B", 0)
        # Assert
        assert pd.isna(result["B"].iloc[0])

    def test_datetime_column_move_preserves_nat_position(self):
        # Arrange
        df = pd.DataFrame(
            {
                "dates": pd.to_datetime(
                    ["2023-01-01", pd.NaT, "2023-01-03"]
                ),
                "values": [1, 2, 3],
            }
        )
        # Act
        result = mv_to_last(df, "dates")
        # Assert
        assert pd.isna(result["dates"].iloc[1])


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__)])
