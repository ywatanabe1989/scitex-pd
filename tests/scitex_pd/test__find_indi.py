#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for scitex_pd.find_indi."""

import numpy as np
import pandas as pd
import pytest

from scitex_pd import find_indi


class TestFindIndiBasic:
    """Test basic functionality of find_indi."""

    def test_single_string_condition_returns_indices_list(self):
        # Arrange
        df = pd.DataFrame({"A": ["x", "y", "x", "z"], "B": [1, 2, 3, 4]})
        conditions = {"A": "x"}
        # Act
        result = find_indi(df, conditions)
        # Assert
        assert isinstance(result, list)

    def test_single_string_condition_matches_expected_rows(self):
        # Arrange
        df = pd.DataFrame({"A": ["x", "y", "x", "z"], "B": [1, 2, 3, 4]})
        conditions = {"A": "x"}
        # Act
        result = find_indi(df, conditions)
        # Assert
        assert result == [0, 2]

    def test_single_numeric_condition_matches_expected_rows(self):
        # Arrange
        df = pd.DataFrame({"A": [1, 2, 3, 2], "B": ["a", "b", "c", "d"]})
        # Act
        result = find_indi(df, {"A": 2})
        # Assert
        assert result == [1, 3]

    def test_multiple_conditions_apply_logical_and(self):
        # Arrange
        df = pd.DataFrame(
            {
                "A": [1, 2, 1, 2],
                "B": ["x", "x", "y", "y"],
                "C": [10, 20, 30, 40],
            }
        )
        # Act
        result = find_indi(df, {"A": 1, "B": "x"})
        # Assert
        assert result == [0]

    def test_list_condition_matches_each_listed_value(self):
        # Arrange
        df = pd.DataFrame(
            {"A": [1, 2, 3, 4, 5], "B": ["a", "b", "c", "d", "e"]}
        )
        # Act
        result = find_indi(df, {"A": [1, 3, 5]})
        # Assert
        assert result == [0, 2, 4]

    def test_mixed_single_and_list_conditions_apply_intersection(self):
        # Arrange
        df = pd.DataFrame(
            {
                "A": [1, 2, 3, 1, 2],
                "B": ["x", "y", "z", "x", "y"],
                "C": [100, 200, 300, 400, 500],
            }
        )
        # Act
        result = find_indi(df, {"A": [1, 2], "B": "x"})
        # Assert
        assert result == [0, 3]


class TestFindIndiNaNHandling:
    """Test NaN handling in find_indi."""

    def test_nan_in_dataframe_skipped_when_not_in_condition(self):
        # Arrange
        df = pd.DataFrame({"A": [1, 2, np.nan, 4], "B": ["x", "y", "z", "w"]})
        # Act
        result = find_indi(df, {"A": 2})
        # Assert
        assert result == [1]

    def test_single_nan_condition_matches_nan_rows(self):
        # Arrange
        df = pd.DataFrame(
            {"A": [1, np.nan, 3, np.nan], "B": ["a", "b", "c", "d"]}
        )
        # Act
        result = find_indi(df, {"A": np.nan})
        # Assert
        assert result == [1, 3]

    def test_nan_in_list_condition_matches_value_or_nan(self):
        # Arrange
        df = pd.DataFrame({"A": [1, np.nan, 3, 4, np.nan]})
        # Act
        result = find_indi(df, {"A": [1, np.nan]})
        # Assert
        assert result == [0, 1, 4]

    def test_none_in_condition_matches_none_rows(self):
        # Arrange
        df = pd.DataFrame(
            {"A": [1, None, 3, None], "B": ["a", "b", "c", "d"]}
        )
        # Act
        result = find_indi(df, {"A": None})
        # Assert
        assert result == [1, 3]

    def test_pd_na_in_condition_matches_pd_na_rows(self):
        # Arrange
        df = pd.DataFrame({"A": [1, pd.NA, 3, pd.NA]}, dtype="Int64")
        # Act
        result = find_indi(df, {"A": pd.NA})
        # Assert
        assert result == [1, 3]


class TestFindIndiEdgeCases:
    """Test edge cases in find_indi."""

    def test_empty_dataframe_and_empty_conditions_return_empty_list(self):
        # Arrange
        df = pd.DataFrame()
        # Act
        result = find_indi(df, {})
        # Assert
        assert result == []

    def test_empty_conditions_on_populated_dataframe_returns_empty_list(self):
        # Arrange
        df = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})
        # Act
        result = find_indi(df, {})
        # Assert
        assert result == []

    def test_no_matching_rows_returns_empty_list(self):
        # Arrange
        df = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})
        # Act
        result = find_indi(df, {"A": 999})
        # Assert
        assert result == []

    def test_all_rows_match_returns_every_index(self):
        # Arrange
        df = pd.DataFrame({"A": [1, 1, 1], "B": ["x", "x", "x"]})
        # Act
        result = find_indi(df, {"A": 1, "B": "x"})
        # Assert
        assert result == [0, 1, 2]

    def test_custom_index_returns_custom_label_for_match(self):
        # Arrange
        df = pd.DataFrame(
            {"A": [1, 2, 3], "B": ["x", "y", "z"]}, index=[10, 20, 30]
        )
        # Act
        result = find_indi(df, {"A": 2})
        # Assert
        assert result == [20]


class TestFindIndiErrorHandling:
    """Test error handling in find_indi."""

    def test_missing_column_raises_keyerror_with_columns_message(self):
        # Arrange
        df = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})
        conditions = {"C": 1}
        # Act
        ctx = pytest.raises(
            KeyError, match=r"Columns not found in DataFrame: \['C'\]"
        )
        # Assert
        with ctx:
            find_indi(df, conditions)

    def test_multiple_missing_columns_raises_keyerror(self):
        # Arrange
        df = pd.DataFrame({"A": [1, 2, 3]})
        conditions = {"B": 1, "C": 2}
        # Act
        ctx = pytest.raises(KeyError, match="Columns not found in DataFrame")
        # Assert
        with ctx:
            find_indi(df, conditions)


class TestFindIndiDataTypes:
    """Test find_indi with various data types."""

    def test_float_value_condition_matches_exact_equality(self):
        # Arrange
        df = pd.DataFrame({"A": [1.1, 2.2, 3.3, 2.2]})
        # Act
        result = find_indi(df, {"A": 2.2})
        # Assert
        assert result == [1, 3]

    def test_boolean_value_condition_matches_true_rows(self):
        # Arrange
        df = pd.DataFrame({"A": [True, False, True, False]})
        # Act
        result = find_indi(df, {"A": True})
        # Assert
        assert result == [0, 2]

    def test_datetime_condition_matches_timestamp_rows(self):
        # Arrange
        dates = pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-01"])
        df = pd.DataFrame({"date": dates})
        # Act
        result = find_indi(df, {"date": pd.Timestamp("2021-01-01")})
        # Assert
        assert result == [0, 2]

    def test_categorical_value_condition_matches_category_rows(self):
        # Arrange
        df = pd.DataFrame(
            {"A": pd.Categorical(["cat", "dog", "cat", "bird"])}
        )
        # Act
        result = find_indi(df, {"A": "cat"})
        # Assert
        assert result == [0, 2]


class TestFindIndiComplexScenarios:
    """Test complex scenarios with find_indi."""

    def test_multi_column_multi_value_conditions_intersect_correctly(self):
        # Arrange
        df = pd.DataFrame(
            {
                "A": [1, 2, 3, 4, 5],
                "B": ["x", "y", "z", "x", "y"],
                "C": [10, 20, 30, 40, 50],
            }
        )
        # Act
        result = find_indi(df, {"A": [1, 2, 3], "B": ["x", "y"]})
        # Assert
        assert result == [0, 1]

    def test_large_dataframe_result_matches_manual_filter(self):
        # Arrange
        np.random.seed(0)
        n = 10000
        df = pd.DataFrame(
            {
                "A": np.random.randint(0, 10, n),
                "B": np.random.choice(["x", "y", "z"], n),
                "C": np.random.rand(n),
            }
        )
        expected = df[(df["A"] == 5) & (df["B"] == "x")].index.tolist()
        # Act
        result = find_indi(df, {"A": 5, "B": "x"})
        # Assert
        assert result == expected

    def test_tuple_condition_behaves_like_list_condition(self):
        # Arrange
        df = pd.DataFrame({"A": [1, 2, 3, 4, 5]})
        # Act
        result = find_indi(df, {"A": (2, 4)})
        # Assert
        assert result == [1, 3]

    def test_mixed_types_in_list_condition_match_each_typed_value(self):
        # Arrange
        df = pd.DataFrame({"A": [1, "2", 3, "4", 5]})
        # Act
        result = find_indi(df, {"A": [1, "2", 3]})
        # Assert
        assert result == [0, 1, 2]


class TestFindIndiDocumentationExamples:
    """Test examples from documentation."""

    def test_docstring_example_matches_documented_indices(self):
        # Arrange
        df = pd.DataFrame({"A": [1, 2, None], "B": ["x", "y", "x"]})
        # Act
        result = find_indi(df, {"A": [1, None], "B": "x"})
        # Assert
        assert result == [0, 2]

    def test_original_commented_example_matches_documented_indices(self):
        # Arrange
        df = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "x"]})
        # Act
        result = find_indi(df, {"A": [1, 2], "B": "x"})
        # Assert
        assert result == [0]


class TestFindIndiNaPaths:
    """NaN-in-list paths + TypeError fallback inside the list branch."""

    def test_pd_na_in_list_is_treated_as_na_match(self):
        # Arrange
        df = pd.DataFrame({"A": pd.array([1, 2, pd.NA], dtype="Int64")})
        # Act
        result = find_indi(df, {"A": [1, pd.NA]})
        # Assert
        assert result == [0, 2]

    def test_typeerror_in_eq_falls_back_to_isna_path(self):
        # Arrange
        # An object whose ``__eq__`` raises forces ``None in value`` to
        # propagate ``TypeError``, exercising the except branch.
        class BadEq:
            def __eq__(self, other):
                raise TypeError("boom")

            def __hash__(self):
                return id(self)

        df = pd.DataFrame({"A": ["x", "y", "z"]})
        # Act
        result = find_indi(df, {"A": ["x", BadEq()]})
        # Assert
        assert result == [0]


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__)])
