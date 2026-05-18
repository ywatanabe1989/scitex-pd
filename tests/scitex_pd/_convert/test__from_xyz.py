#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for scitex_pd.from_xyz."""

import runpy

import numpy as np
import pandas as pd
import pytest

from scitex_pd import from_xyz


class TestBasicFunctionality:
    """Test basic functionality of from_xyz."""

    def test_simple_xyz_conversion_returns_pivot_dataframe(self):
        # Arrange
        data = pd.DataFrame(
            {
                "x": ["A", "B", "C", "A"],
                "y": ["X", "Y", "Z", "Y"],
                "z": [1, 2, 3, 4],
            }
        )
        # Act
        result = from_xyz(data)
        # Assert
        assert isinstance(result, pd.DataFrame)

    def test_simple_xyz_pivot_picks_latest_value_for_repeat(self):
        # Arrange
        data = pd.DataFrame(
            {
                "x": ["A", "B", "C", "A"],
                "y": ["X", "Y", "Z", "Y"],
                "z": [1, 2, 3, 4],
            }
        )
        # Act
        result = from_xyz(data)
        # Assert
        assert result.loc["Y", "A"] == 4

    def test_simple_xyz_pivot_records_third_row(self):
        # Arrange
        data = pd.DataFrame(
            {
                "x": ["A", "B", "C", "A"],
                "y": ["X", "Y", "Z", "Y"],
                "z": [1, 2, 3, 4],
            }
        )
        # Act
        result = from_xyz(data)
        # Assert
        assert result.loc["Z", "C"] == 3

    def test_custom_column_names_yield_expected_pivot_value(self):
        # Arrange
        data = pd.DataFrame(
            {
                "col1": ["A", "B", "C"],
                "col2": ["X", "Y", "Z"],
                "values": [10, 20, 30],
            }
        )
        # Act
        result = from_xyz(data, x="col1", y="col2", z="values")
        # Assert
        assert result.loc["Y", "B"] == 20

    def test_missing_combinations_are_filled_with_zero(self):
        # Arrange
        data = pd.DataFrame({"x": ["A", "B"], "y": ["X", "Y"], "z": [1, 2]})
        # Act
        result = from_xyz(data)
        # Assert
        assert result.loc["X", "B"] == 0

    def test_duplicate_xy_pairs_keep_first_value(self):
        # Arrange
        data = pd.DataFrame(
            {"x": ["A", "A", "A"], "y": ["X", "X", "X"], "z": [1, 2, 3]}
        )
        # Act
        result = from_xyz(data)
        # Assert
        assert result.loc["X", "A"] == 1


class TestSquareMatrix:
    """Test square matrix functionality."""

    def test_square_false_default_returns_non_square_shape(self):
        # Arrange
        data = pd.DataFrame(
            {"x": ["A", "B", "C"], "y": ["X", "Y", "Y"], "z": [1, 2, 3]}
        )
        # Act
        result = from_xyz(data, square=False)
        # Assert
        assert result.shape == (2, 3)

    def test_square_true_returns_square_matrix_shape(self):
        # Arrange
        data = pd.DataFrame(
            {"x": ["A", "B", "C"], "y": ["X", "Y", "Y"], "z": [1, 2, 3]}
        )
        # Act
        result = from_xyz(data, square=True)
        # Assert
        assert result.shape == (5, 5)

    def test_square_true_uses_all_unique_labels_for_index(self):
        # Arrange
        data = pd.DataFrame(
            {"x": ["A", "B", "C"], "y": ["X", "Y", "Y"], "z": [1, 2, 3]}
        )
        # Act
        result = from_xyz(data, square=True)
        # Assert
        assert list(result.index) == ["A", "B", "C", "X", "Y"]

    def test_square_with_identical_labels_returns_three_by_three(self):
        # Arrange
        data = pd.DataFrame(
            {
                "x": ["A", "B", "C", "A"],
                "y": ["B", "C", "A", "C"],
                "z": [1, 2, 3, 4],
            }
        )
        # Act
        result = from_xyz(data, square=True)
        # Assert
        assert result.shape == (3, 3)


class TestDataTypes:
    """Test handling of different data types."""

    def test_numeric_labels_pivot_picks_latest_value_for_duplicate(self):
        # Arrange
        data = pd.DataFrame(
            {
                "x": [1, 2, 3, 1],
                "y": [10, 20, 30, 20],
                "z": [0.1, 0.2, 0.3, 0.4],
            }
        )
        # Act
        result = from_xyz(data)
        # Assert
        assert result.loc[20, 1] == 0.4

    def test_mixed_type_labels_resolve_to_object_pivot(self):
        # Arrange
        data = pd.DataFrame(
            {
                "x": [1, "B", 3.14, 1],
                "y": ["alpha", "beta", "gamma", "beta"],
                "z": [10, 20, 30, 40],
            }
        )
        # Act
        result = from_xyz(data)
        # Assert
        assert result.loc["beta", 1] == 40

    def test_float_z_values_preserve_decimal_precision(self):
        # Arrange
        data = pd.DataFrame(
            {"x": ["A", "B", "C"], "y": ["X", "Y", "Z"], "z": [1.5, 2.7, 3.9]}
        )
        # Act
        result = from_xyz(data)
        # Assert
        assert result.loc["Y", "B"] == 2.7

    def test_string_z_values_keep_first_string_for_present_pair(self):
        # Arrange
        data = pd.DataFrame(
            {"x": ["A", "B"], "y": ["X", "Y"], "z": ["high", "low"]}
        )
        # Act
        result = from_xyz(data)
        # Assert
        assert result.loc["X", "A"] == "high"

    def test_string_z_values_fill_missing_pairs_with_integer_zero(self):
        # Arrange
        data = pd.DataFrame(
            {"x": ["A", "B"], "y": ["X", "Y"], "z": ["high", "low"]}
        )
        # Act
        result = from_xyz(data)
        # Assert
        assert result.loc["X", "B"] == 0


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_dataframe_returns_empty_dataframe(self):
        # Arrange
        data = pd.DataFrame({"x": [], "y": [], "z": []})
        # Act
        result = from_xyz(data)
        # Assert
        assert result.empty

    def test_single_row_returns_one_by_one_matrix_with_value(self):
        # Arrange
        data = pd.DataFrame({"x": ["A"], "y": ["X"], "z": [42]})
        # Act
        result = from_xyz(data)
        # Assert
        assert result.loc["X", "A"] == 42

    def test_missing_columns_raises_keyerror(self):
        # Arrange
        data = pd.DataFrame({"a": [1], "b": [2]})
        # Act
        ctx = pytest.raises(KeyError)
        # Assert
        with ctx:
            from_xyz(data)

    def test_nan_values_drop_rows_during_pivot(self):
        # Arrange
        data = pd.DataFrame(
            {
                "x": ["A", "B", "C"],
                "y": ["X", "Y", "Z"],
                "z": [1, np.nan, 3],
            }
        )
        # Act
        result = from_xyz(data)
        # Assert
        assert "Y" not in result.index

    def test_none_in_labels_drops_corresponding_row(self):
        # Arrange
        data = pd.DataFrame(
            {
                "x": ["A", None, "C"],
                "y": ["X", "Y", None],
                "z": [1, 2, 3],
            }
        )
        # Act
        result = from_xyz(data)
        # Assert
        assert result.shape == (1, 1)


class TestAggregation:
    """Test aggregation behavior."""

    def test_first_aggregation_keeps_first_value_for_duplicates(self):
        # Arrange
        data = pd.DataFrame(
            {"x": ["A", "A", "A"], "y": ["X", "X", "X"], "z": [1, 2, 3]}
        )
        # Act
        result = from_xyz(data)
        # Assert
        assert result.loc["X", "A"] == 1

    def test_multi_duplicate_xy_pairs_keep_first_for_each_pair(self):
        # Arrange
        data = pd.DataFrame(
            {
                "x": ["A", "B", "A", "B", "A"],
                "y": ["X", "Y", "X", "Y", "X"],
                "z": [1, 2, 3, 4, 5],
            }
        )
        # Act
        result = from_xyz(data)
        # Assert
        assert result.loc["Y", "B"] == 2

    def test_order_preservation_returns_sorted_index_labels(self):
        # Arrange
        data = pd.DataFrame(
            {"x": ["C", "B", "A"], "y": ["Z", "Y", "X"], "z": [3, 2, 1]}
        )
        # Act
        result = from_xyz(data)
        # Assert
        assert list(result.index) == ["X", "Y", "Z"]


class TestRealWorldScenarios:
    """Test real-world use cases."""

    def test_statistical_pvalues_matrix_returns_two_by_three(self):
        # Arrange
        data = pd.DataFrame(
            {
                "x": ["gene1", "gene2", "gene3", "gene1", "gene2"],
                "y": [
                    "condition1",
                    "condition1",
                    "condition1",
                    "condition2",
                    "condition2",
                ],
                "z": [0.01, 0.05, 0.001, 0.1, 0.02],
            }
        )
        # Act
        result = from_xyz(data)
        # Assert
        assert result.shape == (2, 3)

    def test_correlation_matrix_construction_returns_three_by_three(self):
        # Arrange
        data = pd.DataFrame(
            {
                "x": ["A", "A", "A", "B", "B", "C"],
                "y": ["A", "B", "C", "B", "C", "C"],
                "z": [1.0, 0.8, 0.6, 1.0, 0.7, 1.0],
            }
        )
        # Act
        result = from_xyz(data, square=True)
        # Assert
        assert result.shape == (3, 3)

    def test_contingency_table_yields_expected_yes_group1_value(self):
        # Arrange
        data = pd.DataFrame(
            {
                "x": ["Yes", "No", "Yes", "No", "Yes"],
                "y": [
                    "Group1",
                    "Group1",
                    "Group2",
                    "Group2",
                    "Group1",
                ],
                "z": [15, 10, 20, 5, 5],
            }
        )
        # Act
        result = from_xyz(data)
        # Assert
        assert result.loc["Group1", "Yes"] == 15


class TestDocstringExample:
    """Test the example from the docstring."""

    def test_docstring_example_pivot_picks_expected_value(self):
        # Arrange
        data = pd.DataFrame(
            {
                "col1": ["A", "B", "C", "A"],
                "col2": ["X", "Y", "Z", "Y"],
                "p_val": [0.01, 0.05, 0.001, 0.1],
            }
        )
        data = data.rename(columns={"col1": "x", "col2": "y", "p_val": "z"})
        # Act
        result = from_xyz(data)
        # Assert
        assert result.loc["X", "A"] == 0.01


class TestLargeDatasets:
    """Test with larger datasets."""

    def test_large_sparse_data_returns_expected_shape(self):
        # Arrange
        np.random.seed(42)
        n_points = 1000
        x_vals = np.random.choice(list("ABCDEFGHIJ"), n_points)
        y_vals = np.random.choice(list("KLMNOPQRST"), n_points)
        z_vals = np.random.rand(n_points)
        data = pd.DataFrame({"x": x_vals, "y": y_vals, "z": z_vals})
        # Act
        result = from_xyz(data)
        # Assert
        assert result.shape == (10, 10)

    def test_dense_categorical_data_produces_no_zero_entries(self):
        # Arrange
        np.random.seed(0)
        x_vals, y_vals, z_vals = [], [], []
        for x in ["A", "B", "C"]:
            for y in ["X", "Y", "Z"]:
                x_vals.extend([x] * 100)
                y_vals.extend([y] * 100)
                z_vals.extend(np.random.rand(100))
        data = pd.DataFrame(
            {
                "x": pd.Categorical(x_vals),
                "y": pd.Categorical(y_vals),
                "z": z_vals,
            }
        )
        # Act
        result = from_xyz(data)
        # Assert
        assert (result != 0).all().all()


class TestFromXyzMainBlock:
    """Run the module-level `__main__` demo via runpy."""

    def test_main_block_prints_to_stdout(self, capsys):
        # Arrange
        module = "scitex_pd._convert._from_xyz"
        # Act
        runpy.run_module(module, run_name="__main__")
        captured = capsys.readouterr()
        # Assert
        assert captured.out


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__)])
