#!/usr/bin/env python3
"""Tests for scitex_pd.sort."""

import numpy as np
import pandas as pd
import pytest

from _helpers import frames_match
from scitex_pd import sort as pd_sort


@pytest.fixture
def sample_df():
    """A small DataFrame with mixed dtypes used across the sort tests."""
    return pd.DataFrame(
        {
            "A": ["foo", "bar", "baz", "qux"],
            "B": [3, 1, 4, 2],
            "C": [2.5, 1.2, 3.8, np.nan],
        }
    )


@pytest.fixture
def df_with_nulls():
    """A DataFrame with null values used to exercise the ``na_position`` knob."""
    return pd.DataFrame({"A": ["a", None, "c", "b"], "B": [1, 2, np.nan, 4]})


class TestSort:
    """Test class for sort function."""

    def test_sort_symbol_is_callable_after_import(self):
        # Arrange
        from scitex_pd import sort as imported
        # Act
        target = imported
        # Assert
        assert callable(target)

    def test_basic_sort_by_column_b_sorts_b_ascending(self, sample_df):
        # Arrange
        df = sample_df
        # Act
        result = pd_sort(df, by="B")
        # Assert
        assert list(result["B"]) == [1, 2, 3, 4]

    def test_basic_sort_by_column_b_reorders_column_a(self, sample_df):
        # Arrange
        df = sample_df
        # Act
        result = pd_sort(df, by="B")
        # Assert
        assert list(result["A"]) == ["bar", "qux", "foo", "baz"]

    def test_sort_descending_returns_b_in_reverse(self, sample_df):
        # Arrange
        df = sample_df
        # Act
        result = pd_sort(df, by="B", ascending=False)
        # Assert
        assert list(result["B"]) == [4, 3, 2, 1]

    def test_sort_descending_reorders_column_a_correspondingly(self, sample_df):
        # Arrange
        df = sample_df
        # Act
        result = pd_sort(df, by="B", ascending=False)
        # Assert
        assert list(result["A"]) == ["baz", "foo", "qux", "bar"]

    def test_sort_multiple_columns_breaks_ties_with_second_column(self):
        # Arrange
        df = pd.DataFrame({"A": [1, 1, 2, 2], "B": [4, 3, 2, 1]})
        # Act
        result = pd_sort(df, by=["A", "B"])
        # Assert
        assert list(result["B"]) == [3, 4, 1, 2]

    def test_sort_with_mixed_ascending_flags_reorders_b_descending_within_a(self):
        # Arrange
        df = pd.DataFrame({"A": [1, 1, 2, 2], "B": [3, 4, 1, 2]})
        # Act
        result = pd_sort(df, by=["A", "B"], ascending=[True, False])
        # Assert
        assert list(result["B"]) == [4, 3, 2, 1]

    def test_na_position_last_puts_nan_at_end_of_column_b(self, df_with_nulls):
        # Arrange
        df = df_with_nulls
        # Act
        result = pd_sort(df, by="B", na_position="last")
        # Assert
        assert pd.isna(result["B"].iloc[-1])

    def test_na_position_first_puts_nan_at_start_of_column_b(self, df_with_nulls):
        # Arrange
        df = df_with_nulls
        # Act
        result = pd_sort(df, by="B", na_position="first")
        # Assert
        assert pd.isna(result["B"].iloc[0])

    def test_ignore_index_false_keeps_original_index_values(self, sample_df):
        # Arrange
        sample_df.index = [10, 20, 30, 40]
        # Act
        result = pd_sort(sample_df, by="B", ignore_index=False)
        # Assert
        assert list(result.index) == [20, 40, 10, 30]

    def test_ignore_index_true_resets_index_to_zero_based_range(self, sample_df):
        # Arrange
        sample_df.index = [10, 20, 30, 40]
        # Act
        result = pd_sort(sample_df, by="B", ignore_index=True)
        # Assert
        assert list(result.index) == [0, 1, 2, 3]

    def test_custom_orders_sort_column_a_by_declared_categories(self):
        # Arrange
        df = pd.DataFrame(
            {
                "A": ["small", "medium", "large", "small", "large"],
                "B": [1, 2, 3, 4, 5],
            }
        )
        custom_order = {"A": ["small", "medium", "large"]}
        # Act
        result = pd_sort(df, orders=custom_order)
        # Assert
        assert result["A"].tolist() == ["small", "small", "medium", "large", "large"]

    def test_multiple_custom_orders_place_size_s_first(self):
        # Arrange
        df = pd.DataFrame(
            {
                "Size": ["L", "S", "M", "L", "S"],
                "Priority": ["high", "low", "medium", "low", "high"],
            }
        )
        custom_order = {
            "Size": ["S", "M", "L"],
            "Priority": ["low", "medium", "high"],
        }
        # Act
        result = pd_sort(df, orders=custom_order)
        # Assert
        assert list(result["Size"][:2]) == ["S", "S"]

    def test_multiple_custom_orders_place_size_l_last(self):
        # Arrange
        df = pd.DataFrame(
            {
                "Size": ["L", "S", "M", "L", "S"],
                "Priority": ["high", "low", "medium", "low", "high"],
            }
        )
        custom_order = {
            "Size": ["S", "M", "L"],
            "Priority": ["low", "medium", "high"],
        }
        # Act
        result = pd_sort(df, orders=custom_order)
        # Assert
        assert list(result["Size"][-2:]) == ["L", "L"]

    def test_inplace_true_returns_same_object_reference(self, sample_df):
        # Arrange
        original_id = id(sample_df)
        # Act
        result = pd_sort(sample_df, by="B", inplace=True)
        # Assert
        assert id(result) == original_id

    def test_inplace_true_does_not_reorder_rows(self, sample_df):
        # Arrange
        # Note: the inplace implementation uses ``update()`` which doesn't
        # reorder rows, so the original order is preserved.
        # Act
        result = pd_sort(sample_df, by="B", inplace=True)
        # Assert
        assert list(result["B"]) == [3, 1, 4, 2]

    def test_sort_by_single_column_moves_sorted_column_to_front(self, sample_df):
        # Arrange
        df = sample_df
        # Act
        result = pd_sort(df, by="B")
        # Assert
        assert list(result.columns) == ["B", "A", "C"]

    def test_sort_by_multi_column_moves_sorted_columns_to_front(self, sample_df):
        # Arrange
        df = sample_df
        # Act
        result = pd_sort(df, by=["C", "B"])
        # Assert
        assert list(result.columns) == ["C", "B", "A"]

    def test_key_function_enables_case_insensitive_sort_on_a(self):
        # Arrange
        df = pd.DataFrame(
            {"A": ["apple", "Banana", "cherry", "Date"], "B": [1, 2, 3, 4]}
        )
        # Act
        result = pd_sort(df, by="A", key=lambda x: x.str.lower())
        # Assert
        assert list(result["A"]) == ["apple", "Banana", "cherry", "Date"]

    @pytest.mark.parametrize(
        "algorithm", ["quicksort", "mergesort", "heapsort", "stable"]
    )
    def test_kind_parameter_supports_documented_algorithms(
        self, sample_df, algorithm
    ):
        # Arrange
        df = sample_df
        # Act
        result = pd_sort(df, by="B", kind=algorithm)
        # Assert
        assert list(result["B"]) == [1, 2, 3, 4]

    def test_empty_dataframe_with_columns_returns_empty_dataframe(self):
        # Arrange
        df = pd.DataFrame(columns=["A", "B"])
        # Act
        result = pd_sort(df, by="A")
        # Assert
        assert result.empty

    def test_empty_dataframe_with_columns_preserves_column_schema(self):
        # Arrange
        df = pd.DataFrame(columns=["A", "B"])
        # Act
        result = pd_sort(df, by="A")
        # Assert
        assert list(result.columns) == ["A", "B"]

    def test_single_row_dataframe_returns_input_unchanged(self):
        # Arrange
        df = pd.DataFrame({"A": [1], "B": [2]})
        # Act
        result = pd_sort(df, by="A")
        # Assert
        assert frames_match(result, df)

    def test_sort_with_orders_only_orders_by_provided_categories(self, sample_df):
        # Arrange
        orders = {"A": ["bar", "baz", "foo", "qux"]}
        # Act
        result = pd_sort(sample_df, orders=orders)
        # Assert
        assert list(result["A"]) == ["bar", "baz", "foo", "qux"]

    def test_sort_by_nonexistent_column_raises_keyerror(self):
        # Arrange
        df = pd.DataFrame({"A": [1, 2, 3]})
        # Act
        ctx = pytest.raises(KeyError)
        # Assert
        with ctx:
            pd_sort(df, by="NonExistent")

    @pytest.mark.parametrize(
        "input_value", [[1, 2, 3], "not a dataframe", 123]
    )
    def test_non_dataframe_input_raises_attribute_error(self, input_value):
        # Arrange
        bad = input_value
        # Act
        ctx = pytest.raises(AttributeError)
        # Assert
        with ctx:
            pd_sort(bad, by="A")


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__)])
