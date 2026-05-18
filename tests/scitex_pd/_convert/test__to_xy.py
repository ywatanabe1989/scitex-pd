#!/usr/bin/env python3
"""Tests for scitex_pd.to_xy."""

import numpy as np
import pandas as pd
import pytest

from scitex_pd import to_xy


@pytest.fixture
def square_df():
    """A square DataFrame used across the suite."""
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    return pd.DataFrame(data, index=["A", "B", "C"], columns=["A", "B", "C"])


@pytest.fixture
def numeric_index_df():
    """A DataFrame with numeric index and named columns."""
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    return pd.DataFrame(data, index=[0, 1, 2], columns=["A", "B", "C"])


@pytest.fixture
def numeric_columns_df():
    """A DataFrame with named index and numeric columns."""
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    return pd.DataFrame(data, index=["A", "B", "C"], columns=[0, 1, 2])


class TestToXY:
    """Test class for to_xy function."""

    def test_to_xy_symbol_is_callable_after_import(self):
        # Arrange
        from scitex_pd import to_xy as imported
        # Act
        target = imported
        # Assert
        assert callable(target)

    def test_basic_conversion_returns_nine_by_three_dataframe(self, square_df):
        # Arrange
        df = square_df
        # Act
        result = to_xy(df)
        # Assert
        assert result.shape == (9, 3)

    def test_basic_conversion_uses_xyz_column_names(self, square_df):
        # Arrange
        df = square_df
        # Act
        result = to_xy(df)
        # Assert
        assert list(result.columns) == ["x", "y", "z"]

    def test_basic_conversion_preserves_first_row_value(self, square_df):
        # Arrange
        df = square_df
        # Act
        result = to_xy(df)
        # Assert
        assert (
            result.iloc[0]["x"],
            result.iloc[0]["y"],
            result.iloc[0]["z"],
        ) == ("A", "A", 1)

    def test_numeric_index_collapses_to_unique_numeric_x_labels(
        self, numeric_index_df
    ):
        # Arrange
        df = numeric_index_df
        # Act
        result = to_xy(df)
        # Assert
        assert sorted(result["x"].unique()) == [0, 1, 2]

    def test_numeric_index_collapses_to_unique_numeric_y_labels(
        self, numeric_index_df
    ):
        # Arrange
        df = numeric_index_df
        # Act
        result = to_xy(df)
        # Assert
        assert sorted(result["y"].unique()) == [0, 1, 2]

    def test_numeric_columns_collapse_to_unique_numeric_y_labels(
        self, numeric_columns_df
    ):
        # Arrange
        df = numeric_columns_df
        # Act
        result = to_xy(df)
        # Assert
        assert sorted(result["y"].unique()) == [0, 1, 2]

    def test_non_square_dataframe_raises_assertion_error(self):
        # Arrange
        df = pd.DataFrame(np.array([[1, 2], [3, 4], [5, 6]]))
        # Act
        ctx = pytest.raises(AssertionError)
        # Assert
        with ctx:
            to_xy(df)

    def test_identity_matrix_diagonal_entries_are_one(self):
        # Arrange
        data = np.eye(3)
        df = pd.DataFrame(data, index=["A", "B", "C"], columns=["A", "B", "C"])
        # Act
        result = to_xy(df)
        diagonal = result[result["x"] == result["y"]]
        # Assert
        assert (diagonal["z"] == 1.0).all()

    def test_identity_matrix_off_diagonal_entries_are_zero(self):
        # Arrange
        data = np.eye(3)
        df = pd.DataFrame(data, index=["A", "B", "C"], columns=["A", "B", "C"])
        # Act
        result = to_xy(df)
        off_diag = result[result["x"] != result["y"]]
        # Assert
        assert (off_diag["z"] == 0.0).all()

    def test_single_element_dataframe_returns_one_by_three(self):
        # Arrange
        df = pd.DataFrame([[42]], index=["A"], columns=["A"])
        # Act
        result = to_xy(df)
        # Assert
        assert result.shape == (1, 3)

    def test_single_element_dataframe_first_z_value_is_input_value(self):
        # Arrange
        df = pd.DataFrame([[42]], index=["A"], columns=["A"])
        # Act
        result = to_xy(df)
        # Assert
        assert result.iloc[0]["z"] == 42

    def test_nan_values_are_preserved_in_z_column(self):
        # Arrange
        data = np.array(
            [[1, np.nan, 3], [4, 5, np.nan], [np.nan, 8, 9]]
        )
        df = pd.DataFrame(data, index=["A", "B", "C"], columns=["A", "B", "C"])
        # Act
        result = to_xy(df)
        # Assert
        assert result["z"].isna().sum() == 3

    def test_column_order_preserved_when_grouping_by_y_unsorted(
        self, square_df
    ):
        # Arrange
        df = square_df
        # Act
        result = to_xy(df)
        y_values = [
            group["y"].iloc[0]
            for _, group in result.groupby("y", sort=False)
        ]
        # Assert
        assert y_values == ["A", "B", "C"]

    def test_index_order_preserved_for_first_column_block(self, square_df):
        # Arrange
        df = square_df
        # Act
        result = to_xy(df)
        # Assert
        assert list(result.iloc[:3]["x"]) == ["A", "B", "C"]

    def test_duplicate_index_names_raise_attribute_error(self):
        # Arrange
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        df = pd.DataFrame(data, index=["A", "A", "B"], columns=["A", "A", "B"])
        # Act
        ctx = pytest.raises(AttributeError)
        # Assert
        with ctx:
            to_xy(df)

    def test_mismatched_non_numeric_labels_yields_nine_row_output(self):
        # Arrange
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        df = pd.DataFrame(data, index=["A", "B", "C"], columns=["X", "Y", "Z"])
        # Act
        result = to_xy(df)
        # Assert
        assert result.shape == (9, 3)

    def test_mismatched_labels_keep_distinct_x_and_y_label_sets(self):
        # Arrange
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        df = pd.DataFrame(data, index=["A", "B", "C"], columns=["X", "Y", "Z"])
        # Act
        result = to_xy(df)
        # Assert
        assert set(result["x"].unique()) == {"A", "B", "C"}

    @pytest.mark.parametrize(
        "dtype", [int, float, np.float32, np.float64]
    )
    def test_different_dtypes_preserve_numeric_z_column(self, dtype):
        # Arrange
        data = np.array([[1, 2], [3, 4]], dtype=dtype)
        df = pd.DataFrame(data, index=["A", "B"], columns=["A", "B"])
        # Act
        result = to_xy(df)
        # Assert
        assert pd.api.types.is_numeric_dtype(result["z"])


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__)])
