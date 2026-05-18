#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for scitex_pd.melt_cols."""

import numpy as np
import pandas as pd
import pytest

from scitex_pd import melt_cols


class TestBasicFunctionality:
    """Test basic functionality of melt_cols."""

    def test_simple_melt_returns_expected_row_count(self):
        # Arrange
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["A", "B", "C"],
                "score_1": [10, 20, 30],
                "score_2": [15, 25, 35],
            }
        )
        # Act
        result = melt_cols(df, cols=["score_1", "score_2"])
        # Assert
        assert len(result) == 6

    def test_simple_melt_keeps_id_and_value_columns(self):
        # Arrange
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["A", "B", "C"],
                "score_1": [10, 20, 30],
                "score_2": [15, 25, 35],
            }
        )
        # Act
        result = melt_cols(df, cols=["score_1", "score_2"])
        # Assert
        assert {"variable", "value", "id", "name"}.issubset(result.columns)

    def test_simple_melt_first_row_has_expected_score_value(self):
        # Arrange
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["A", "B", "C"],
                "score_1": [10, 20, 30],
                "score_2": [15, 25, 35],
            }
        )
        # Act
        result = melt_cols(df, cols=["score_1", "score_2"])
        # Assert
        assert result.iloc[0]["value"] == 10

    def test_single_column_melt_returns_two_rows(self):
        # Arrange
        df = pd.DataFrame({"id": [1, 2], "value": [100, 200]})
        # Act
        result = melt_cols(df, cols=["value"])
        # Assert
        assert len(result) == 2

    def test_single_column_named_value_renames_to_melted_value(self):
        # Arrange
        df = pd.DataFrame({"id": [1, 2], "value": [100, 200]})
        # Act
        result = melt_cols(df, cols=["value"])
        # Assert
        assert list(result["melted_value"]) == [100, 200]

    def test_multiple_id_columns_preserves_all_identifier_columns(self):
        # Arrange
        df = pd.DataFrame(
            {
                "year": [2020, 2021, 2022],
                "month": [1, 2, 3],
                "category": ["A", "B", "C"],
                "sales": [100, 200, 300],
                "costs": [50, 100, 150],
            }
        )
        # Act
        result = melt_cols(df, cols=["sales", "costs"])
        # Assert
        assert {"year", "month", "category"}.issubset(result.columns)

    def test_multiple_id_columns_preserves_sales_values_in_order(self):
        # Arrange
        df = pd.DataFrame(
            {
                "year": [2020, 2021, 2022],
                "sales": [100, 200, 300],
                "costs": [50, 100, 150],
            }
        )
        # Act
        result = melt_cols(df, cols=["sales", "costs"])
        sales_rows = result[result["variable"] == "sales"]
        # Assert
        assert list(sales_rows["value"]) == [100, 200, 300]


class TestIdColumnParameter:
    """Test id_columns parameter functionality."""

    def test_explicit_id_columns_drops_unspecified_columns(self):
        # Arrange
        df = pd.DataFrame(
            {
                "id": [1, 2],
                "name": ["A", "B"],
                "extra": ["X", "Y"],
                "val1": [10, 20],
                "val2": [30, 40],
            }
        )
        # Act
        result = melt_cols(df, cols=["val1", "val2"], id_columns=["id", "name"])
        # Assert
        assert "extra" not in result.columns

    def test_empty_id_columns_returns_only_variable_and_value(self):
        # Arrange
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        # Act
        result = melt_cols(df, cols=["a", "b"], id_columns=[])
        # Assert
        assert set(result.columns) == {"variable", "value"}

    def test_auto_id_columns_uses_non_melted_columns(self):
        # Arrange
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "group": ["A", "B", "C"],
                "metric1": [10, 20, 30],
                "metric2": [40, 50, 60],
                "metric3": [70, 80, 90],
            }
        )
        # Act
        result = melt_cols(df, cols=["metric1", "metric2", "metric3"])
        # Assert
        assert {"id", "group"}.issubset(result.columns) and len(result) == 9


class TestDataTypes:
    """Test handling of different data types."""

    def test_mixed_data_types_returns_expected_row_count(self):
        # Arrange
        df = pd.DataFrame(
            {
                "id": [1, 2],
                "int_col": [10, 20],
                "float_col": [1.5, 2.5],
                "str_col": ["a", "b"],
                "bool_col": [True, False],
            }
        )
        # Act
        result = melt_cols(df, cols=["int_col", "float_col", "str_col", "bool_col"])
        # Assert
        assert len(result) == 8

    def test_mixed_data_types_value_column_carries_each_input(self):
        # Arrange
        df = pd.DataFrame(
            {
                "id": [1, 2],
                "int_col": [10, 20],
                "float_col": [1.5, 2.5],
                "str_col": ["a", "b"],
            }
        )
        # Act
        result = melt_cols(df, cols=["int_col", "float_col", "str_col"])
        values = set(result["value"].tolist())
        # Assert
        assert {10, 20, 1.5, 2.5, "a", "b"}.issubset(values)

    def test_datetime_columns_value_dtype_remains_datetime(self):
        # Arrange
        df = pd.DataFrame(
            {
                "id": [1, 2],
                "date1": pd.to_datetime(["2023-01-01", "2023-01-02"]),
                "date2": pd.to_datetime(["2023-02-01", "2023-02-02"]),
            }
        )
        # Act
        result = melt_cols(df, cols=["date1", "date2"])
        # Assert
        assert pd.api.types.is_datetime64_any_dtype(result["value"])

    def test_categorical_columns_value_set_equals_input_categories(self):
        # Arrange
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "cat1": pd.Categorical(["A", "B", "C"]),
                "cat2": pd.Categorical(["X", "Y", "Z"]),
            }
        )
        # Act
        result = melt_cols(df, cols=["cat1", "cat2"])
        # Assert
        assert set(result["value"]) == {"A", "B", "C", "X", "Y", "Z"}


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_missing_columns_raises_valueerror(self):
        # Arrange
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        # Act
        ctx = pytest.raises(ValueError, match="Columns not found")
        # Assert
        with ctx:
            melt_cols(df, cols=["c", "d"])

    def test_partial_missing_columns_message_lists_only_missing(self):
        # Arrange
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        # Act
        ctx = pytest.raises(ValueError, match="Columns not found.*{'d'}")
        # Assert
        with ctx:
            melt_cols(df, cols=["a", "b", "d"])

    def test_empty_dataframe_raises_valueerror_for_missing_columns(self):
        # Arrange
        df = pd.DataFrame()
        # Act
        ctx = pytest.raises(ValueError, match="Columns not found")
        # Assert
        with ctx:
            melt_cols(df, cols=["a"])

    def test_single_row_input_returns_two_melted_rows(self):
        # Arrange
        df = pd.DataFrame({"id": [1], "val1": [10], "val2": [20]})
        # Act
        result = melt_cols(df, cols=["val1", "val2"])
        # Assert
        assert list(result["value"]) == [10, 20]

    def test_all_columns_melted_returns_only_variable_and_value(self):
        # Arrange
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        # Act
        result = melt_cols(df, cols=["a", "b", "c"])
        # Assert
        assert set(result.columns) == {"variable", "value"}


class TestNullHandling:
    """Test handling of null values."""

    def test_null_in_melted_columns_preserves_nan_count(self):
        # Arrange
        df = pd.DataFrame(
            {"id": [1, 2, 3], "val1": [10, np.nan, 30], "val2": [np.nan, 20, np.nan]}
        )
        # Act
        result = melt_cols(df, cols=["val1", "val2"])
        # Assert
        assert result["value"].isna().sum() == 3

    def test_null_in_id_columns_preserves_nan_at_expected_row(self):
        # Arrange
        df = pd.DataFrame(
            {"id": [1, np.nan, 3], "name": ["A", "B", None], "value": [10, 20, 30]}
        )
        # Act
        result = melt_cols(df, cols=["value"])
        # Assert
        assert pd.isna(result.iloc[1]["id"])


class TestIndexHandling:
    """Test DataFrame index handling."""

    def test_non_default_index_is_reset_to_range_index(self):
        # Arrange
        df = pd.DataFrame(
            {"val1": [10, 20, 30], "val2": [40, 50, 60]}, index=["a", "b", "c"]
        )
        # Act
        result = melt_cols(df, cols=["val1", "val2"])
        # Assert
        assert result.index.tolist() == list(range(6))

    def test_multiindex_input_yields_range_indexed_output(self):
        # Arrange
        arrays = [["A", "A", "B", "B"], [1, 2, 1, 2]]
        index = pd.MultiIndex.from_arrays(arrays, names=("letter", "number"))
        df = pd.DataFrame(
            {"val1": [10, 20, 30, 40], "val2": [50, 60, 70, 80]}, index=index
        )
        # Act
        result = melt_cols(df, cols=["val1", "val2"])
        # Assert
        assert isinstance(result.index, pd.RangeIndex)


class TestOrderPreservation:
    """Test order preservation in results."""

    def test_row_order_preserved_for_first_variable(self):
        # Arrange
        df = pd.DataFrame(
            {
                "id": [3, 1, 2],
                "name": ["C", "A", "B"],
                "val1": [30, 10, 20],
                "val2": [60, 40, 50],
            }
        )
        # Act
        result = melt_cols(df, cols=["val1", "val2"])
        val1_rows = result[result["variable"] == "val1"]
        # Assert
        assert list(val1_rows["value"]) == [30, 10, 20]

    def test_column_order_keeps_supplied_id_columns_in_output(self):
        # Arrange
        df = pd.DataFrame(
            {"z": [1, 2], "a": [3, 4], "val1": [5, 6], "val2": [7, 8]}
        )
        # Act
        result = melt_cols(df, cols=["val1", "val2"], id_columns=["a", "z"])
        # Assert
        assert {"a", "z", "variable", "value"}.issubset(result.columns)


class TestRealWorldScenarios:
    """Test real-world use cases."""

    def test_time_series_reshape_returns_nine_records(self):
        # Arrange
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=3),
                "location": ["NYC", "LA", "CHI"],
                "temp_morning": [32, 65, 40],
                "temp_afternoon": [45, 78, 55],
                "temp_evening": [38, 70, 42],
            }
        )
        # Act
        result = melt_cols(
            df, cols=["temp_morning", "temp_afternoon", "temp_evening"]
        )
        # Assert
        assert len(result) == 9

    def test_time_series_reshape_groups_three_periods_per_location(self):
        # Arrange
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=3),
                "location": ["NYC", "LA", "CHI"],
                "temp_morning": [32, 65, 40],
                "temp_afternoon": [45, 78, 55],
                "temp_evening": [38, 70, 42],
            }
        )
        # Act
        result = melt_cols(
            df, cols=["temp_morning", "temp_afternoon", "temp_evening"]
        )
        nyc = result[result["location"] == "NYC"]
        # Assert
        assert set(nyc["variable"]) == {
            "temp_morning",
            "temp_afternoon",
            "temp_evening",
        }

    def test_survey_reshape_preserves_respondent_info_columns(self):
        # Arrange
        df = pd.DataFrame(
            {
                "respondent_id": [1, 2, 3],
                "age": [25, 35, 45],
                "gender": ["M", "F", "M"],
                "q1_satisfaction": [4, 5, 3],
                "q2_satisfaction": [5, 4, 4],
                "q3_satisfaction": [3, 5, 2],
            }
        )
        # Act
        result = melt_cols(
            df, cols=["q1_satisfaction", "q2_satisfaction", "q3_satisfaction"]
        )
        # Assert
        assert {"respondent_id", "age", "gender"}.issubset(result.columns)

    def test_survey_reshape_preserves_respondent_answer_order(self):
        # Arrange
        df = pd.DataFrame(
            {
                "respondent_id": [1, 2, 3],
                "q1_satisfaction": [4, 5, 3],
                "q2_satisfaction": [5, 4, 4],
                "q3_satisfaction": [3, 5, 2],
            }
        )
        # Act
        result = melt_cols(
            df, cols=["q1_satisfaction", "q2_satisfaction", "q3_satisfaction"]
        )
        resp1 = result[result["respondent_id"] == 1]
        # Assert
        assert list(resp1["value"]) == [4, 5, 3]


class TestDocstringExample:
    """Test the example from the docstring."""

    def test_docstring_example_returns_six_rows(self):
        # Arrange
        data = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["Alice", "Bob", "Charlie"],
                "score_1": [85, 90, 78],
                "score_2": [92, 88, 95],
            }
        )
        # Act
        melted = melt_cols(data, cols=["score_1", "score_2"])
        # Assert
        assert len(melted) == 6

    def test_docstring_example_columns_match_expected_schema(self):
        # Arrange
        data = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["Alice", "Bob", "Charlie"],
                "score_1": [85, 90, 78],
                "score_2": [92, 88, 95],
            }
        )
        # Act
        melted = melt_cols(data, cols=["score_1", "score_2"])
        # Assert
        assert set(melted.columns) == {"id", "name", "variable", "value"}

    def test_docstring_example_first_score_row_matches_expected_value(self):
        # Arrange
        data = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["Alice", "Bob", "Charlie"],
                "score_1": [85, 90, 78],
                "score_2": [92, 88, 95],
            }
        )
        # Act
        melted = melt_cols(data, cols=["score_1", "score_2"])
        # Assert
        assert melted.iloc[0]["value"] == 85

    def test_docstring_example_fourth_score_row_matches_expected_value(self):
        # Arrange
        data = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["Alice", "Bob", "Charlie"],
                "score_1": [85, 90, 78],
                "score_2": [92, 88, 95],
            }
        )
        # Act
        melted = melt_cols(data, cols=["score_1", "score_2"])
        # Assert
        assert melted.iloc[3]["value"] == 92


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__)])
