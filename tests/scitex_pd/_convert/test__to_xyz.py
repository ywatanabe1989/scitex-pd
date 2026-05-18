#!/usr/bin/env python3
"""Tests for scitex_pd.to_xyz."""

import numpy as np
import pandas as pd
import pytest

from scitex_pd import to_xyz


@pytest.fixture
def rectangular_df():
    """A rectangular DataFrame used across the suite."""
    data = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    return pd.DataFrame(data, index=["A", "B", "C"], columns=["W", "X", "Y", "Z"])


@pytest.fixture
def named_axes_df():
    """A DataFrame with explicit ``index.name`` / ``columns.name``."""
    data = np.array([[1, 2], [3, 4]])
    df = pd.DataFrame(data, index=["row1", "row2"], columns=["col1", "col2"])
    df.index.name = "rows"
    df.columns.name = "cols"
    return df


@pytest.fixture
def numeric_df():
    """A DataFrame with default numeric index and columns."""
    data = np.array([[10, 20, 30], [40, 50, 60]])
    return pd.DataFrame(data)


class TestToXYZ:
    """Test class for to_xyz function."""

    def test_to_xyz_symbol_is_callable_after_import(self):
        # Arrange
        from scitex_pd import to_xyz as imported
        # Act
        target = imported
        # Assert
        assert callable(target)

    def test_basic_conversion_returns_twelve_by_three_dataframe(
        self, rectangular_df
    ):
        # Arrange
        df = rectangular_df
        # Act
        result = to_xyz(df)
        # Assert
        assert result.shape == (12, 3)

    def test_basic_conversion_uses_xyz_column_names_when_unnamed_axes(
        self, rectangular_df
    ):
        # Arrange
        df = rectangular_df
        # Act
        result = to_xyz(df)
        # Assert
        assert list(result.columns) == ["x", "y", "z"]

    def test_basic_conversion_preserves_first_value_triple(
        self, rectangular_df
    ):
        # Arrange
        df = rectangular_df
        # Act
        result = to_xyz(df)
        # Assert
        assert (
            result.iloc[0]["x"],
            result.iloc[0]["y"],
            result.iloc[0]["z"],
        ) == ("A", "W", 1)

    def test_named_axes_use_axis_names_as_xyz_columns(self, named_axes_df):
        # Arrange
        df = named_axes_df
        # Act
        result = to_xyz(df)
        # Assert
        assert list(result.columns) == ["rows", "cols", "z"]

    def test_named_axes_preserves_first_named_row_value(self, named_axes_df):
        # Arrange
        df = named_axes_df
        # Act
        result = to_xyz(df)
        # Assert
        assert result.iloc[0]["z"] == 1

    def test_numeric_indices_use_xyz_column_names(self, numeric_df):
        # Arrange
        df = numeric_df
        # Act
        result = to_xyz(df)
        # Assert
        assert list(result.columns) == ["x", "y", "z"]

    def test_numeric_indices_preserve_first_z_value(self, numeric_df):
        # Arrange
        df = numeric_df
        # Act
        result = to_xyz(df)
        # Assert
        assert result["z"].iloc[0] == 10

    def test_single_column_dataframe_y_values_are_constant_column_name(self):
        # Arrange
        df = pd.DataFrame({"A": [1, 2, 3]}, index=["x1", "x2", "x3"])
        # Act
        result = to_xyz(df)
        # Assert
        assert list(result["y"]) == ["A", "A", "A"]

    def test_single_row_dataframe_x_values_are_constant_index_value(self):
        # Arrange
        df = pd.DataFrame(
            [[1, 2, 3]], columns=["A", "B", "C"], index=["row1"]
        )
        # Act
        result = to_xyz(df)
        # Assert
        assert list(result["x"]) == ["row1", "row1", "row1"]

    def test_nan_values_preserved_in_z_column_count_matches(self):
        # Arrange
        df = pd.DataFrame(
            {
                "A": [1, np.nan, 3],
                "B": [np.nan, 5, 6],
                "C": [7, 8, np.nan],
            },
            index=["X", "Y", "Z"],
        )
        # Act
        result = to_xyz(df)
        # Assert
        assert result["z"].isna().sum() == 3

    def test_empty_dataframe_without_columns_raises_value_error(self):
        # Arrange
        df = pd.DataFrame()
        # Act
        ctx = pytest.raises(ValueError, match="No objects to concatenate")
        # Assert
        with ctx:
            to_xyz(df)

    def test_empty_dataframe_with_columns_returns_empty_xyz_frame(self):
        # Arrange
        df = pd.DataFrame(columns=["A", "B"], index=[])
        # Act
        result = to_xyz(df)
        # Assert
        assert list(result.columns) == ["x", "y", "z"] and result.empty

    def test_column_order_is_preserved_in_y_unique_values(self, rectangular_df):
        # Arrange
        df = rectangular_df
        # Act
        result = to_xyz(df)
        # Assert
        assert list(result["y"].unique()) == ["W", "X", "Y", "Z"]

    def test_string_index_values_appear_in_x_column(self):
        # Arrange
        df = pd.DataFrame(
            {"col1": [100, 200], "col2": [300, 400]},
            index=["first", "second"],
        )
        # Act
        result = to_xyz(df)
        # Assert
        assert sorted(result["x"].unique()) == ["first", "second"]

    def test_mixed_dtypes_preserve_int_column_values_in_z(self):
        # Arrange
        df = pd.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.1, 2.2, 3.3],
                "str_col": ["a", "b", "c"],
            },
            index=["r1", "r2", "r3"],
        )
        # Act
        result = to_xyz(df)
        int_vals = result[result["y"] == "int_col"]["z"].tolist()
        # Assert
        assert int_vals == [1, 2, 3]

    def test_mixed_dtypes_preserve_str_column_values_in_z(self):
        # Arrange
        df = pd.DataFrame(
            {
                "int_col": [1, 2, 3],
                "str_col": ["a", "b", "c"],
            },
            index=["r1", "r2", "r3"],
        )
        # Act
        result = to_xyz(df)
        str_vals = result[result["y"] == "str_col"]["z"].tolist()
        # Assert
        assert str_vals == ["a", "b", "c"]

    def test_multiindex_input_yields_tuple_x_values(self):
        # Arrange
        arrays = [["A", "A", "B", "B"], [1, 2, 1, 2]]
        index = pd.MultiIndex.from_arrays(
            arrays, names=["first", "second"]
        )
        df = pd.DataFrame({"col": [10, 20, 30, 40]}, index=index)
        # Act
        result = to_xyz(df)
        # Assert
        assert isinstance(result["x"].iloc[0], tuple)

    def test_datetime_index_z_dtype_is_datetime_when_propagated(self):
        # Arrange
        dates = pd.date_range("2021-01-01", periods=3)
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}, index=dates)
        # Act
        result = to_xyz(df)
        # Assert
        assert pd.api.types.is_datetime64_any_dtype(result["x"])

    @pytest.mark.parametrize(
        "nrows,ncols", [(1, 10), (10, 1), (5, 5), (3, 7)]
    )
    def test_various_shape_inputs_produce_expected_row_count(
        self, nrows, ncols
    ):
        # Arrange
        data = np.arange(nrows * ncols).reshape(nrows, ncols)
        df = pd.DataFrame(data)
        # Act
        result = to_xyz(df)
        # Assert
        assert result.shape == (nrows * ncols, 3)

    def test_non_square_input_is_supported_without_assertion(self):
        # Arrange
        df = pd.DataFrame(np.arange(12).reshape(3, 4))
        # Act
        result = to_xyz(df)
        # Assert
        assert result.shape == (12, 3)


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__)])
