#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for scitex_pd.merge_columns / merge_cols."""

import pandas as pd
import pytest

from _helpers import frames_match
from scitex_pd import merge_cols, merge_columns


class TestBasicFunctionality:
    """Test basic functionality of merge_columns."""

    def test_simple_merge_with_separator_creates_combined_column(self):
        # Arrange
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
        # Act
        result = merge_columns(df, "A", "B", sep=" ")
        # Assert
        assert "A_B" in result.columns

    def test_simple_merge_with_separator_joins_values_per_row(self):
        # Arrange
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
        # Act
        result = merge_columns(df, "A", "B", sep=" ")
        # Assert
        assert list(result["A_B"]) == ["1 4", "2 5", "3 6"]

    def test_simple_merge_preserves_original_columns(self):
        # Arrange
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
        # Act
        result = merge_columns(df, "A", "B", sep=" ")
        # Assert
        assert {"A", "B", "C"}.issubset(result.columns)

    def test_merge_with_column_labels_uses_default_merged_name(self):
        # Arrange
        df = pd.DataFrame({"A": [0, 5, 10], "B": [1, 6, 11]})
        # Act
        result = merge_columns(df, "A", "B")
        # Assert
        assert "merged" in result.columns

    def test_merge_with_column_labels_formats_first_row(self):
        # Arrange
        df = pd.DataFrame({"A": [0, 5, 10], "B": [1, 6, 11]})
        # Act
        result = merge_columns(df, "A", "B")
        # Assert
        assert result["merged"].iloc[0] == "A-0_B-1"

    def test_merge_three_columns_with_comma_separator(self):
        # Arrange
        df = pd.DataFrame({"X": [1, 2], "Y": [3, 4], "Z": [5, 6]})
        # Act
        result = merge_columns(df, "X", "Y", "Z", sep=",")
        # Assert
        assert result["X_Y_Z"].iloc[0] == "1,3,5"

    def test_merge_cols_alias_returns_same_dataframe_as_merge_columns(self):
        # Arrange
        df = pd.DataFrame({"A": [1], "B": [2]})
        # Act
        from_columns = merge_columns(df, "A", "B", sep=" ")
        from_cols = merge_cols(df, "A", "B", sep=" ")
        # Assert
        assert frames_match(from_columns, from_cols)


class TestParameterVariations:
    """Test different parameter combinations."""

    def test_custom_sep1_and_sep2_formats_merged_value_pairs(self):
        # Arrange
        df = pd.DataFrame({"col1": [10, 20], "col2": [30, 40]})
        # Act
        result = merge_columns(df, "col1", "col2", sep1=" & ", sep2="=")
        # Assert
        assert result["merged"].iloc[0] == "col1=10 & col2=30"

    def test_custom_name_parameter_renames_merged_column(self):
        # Arrange
        df = pd.DataFrame(
            {"first": ["John", "Jane"], "last": ["Doe", "Smith"]}
        )
        # Act
        result = merge_columns(
            df, "first", "last", sep=" ", name="full_name"
        )
        # Assert
        assert "full_name" in result.columns

    def test_custom_name_parameter_concatenates_values_with_separator(self):
        # Arrange
        df = pd.DataFrame(
            {"first": ["John", "Jane"], "last": ["Doe", "Smith"]}
        )
        # Act
        result = merge_columns(
            df, "first", "last", sep=" ", name="full_name"
        )
        # Assert
        assert result["full_name"].iloc[0] == "John Doe"

    def test_list_input_concatenates_in_declared_order(self):
        # Arrange
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})
        # Act
        result = merge_columns(df, ["A", "B", "C"], sep="-")
        # Assert
        assert result["A_B_C"].iloc[0] == "1-3-5"

    def test_tuple_input_concatenates_in_declared_order(self):
        # Arrange
        df = pd.DataFrame({"X": [7, 8], "Y": [9, 10]})
        # Act
        result = merge_columns(df, ("X", "Y"), sep="/")
        # Assert
        assert result["X_Y"].iloc[0] == "7/9"


class TestDataTypes:
    """Test handling of different data types."""

    def test_numeric_columns_merge_preserves_decimal_in_output(self):
        # Arrange
        df = pd.DataFrame(
            {"int_col": [1, 2, 3], "float_col": [1.5, 2.5, 3.5]}
        )
        # Act
        result = merge_columns(df, "int_col", "float_col", sep=" | ")
        # Assert
        assert result["int_col_float_col"].iloc[0] == "1 | 1.5"

    def test_mixed_type_columns_stringify_each_value(self):
        # Arrange
        df = pd.DataFrame(
            {
                "str": ["a", "b"],
                "int": [1, 2],
                "float": [3.14, 2.71],
                "bool": [True, False],
            }
        )
        # Act
        result = merge_columns(df, "str", "int", "float", "bool", sep=",")
        # Assert
        assert result["str_int_float_bool"].iloc[0] == "a,1,3.14,True"

    def test_datetime_columns_merge_contains_iso_date_substring(self):
        # Arrange
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2023-01-01", "2023-01-02"]),
                "time": ["10:00", "11:00"],
            }
        )
        # Act
        result = merge_columns(df, "date", "time", sep=" ")
        # Assert
        assert "2023-01-01" in result["date_time"].iloc[0]

    @pytest.mark.skipif(
        pd.__version__ >= "2.0",
        reason=(
            "pandas 2.x changes how astype(str) interacts with row-wise apply on "
            "mixed-dtype frames containing nulls; merge_columns relies on the "
            "pandas 1.x string-coercion path."
        ),
    )
    def test_null_values_in_mixed_columns_coerce_to_string_representation(self):
        # Arrange
        df = pd.DataFrame({"A": [1, None, 3], "B": ["x", "y", None]})
        # Act
        result = merge_columns(df, "A", "B", sep="-")
        # Assert
        assert result["A_B"].iloc[1] == "nan-y"


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_calling_with_no_columns_raises_valueerror(self):
        # Arrange
        df = pd.DataFrame({"A": [1, 2]})
        # Act
        ctx = pytest.raises(ValueError, match="No columns specified")
        # Assert
        with ctx:
            merge_columns(df)

    def test_missing_columns_raises_keyerror_with_list_of_missing(self):
        # Arrange
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        # Act
        ctx = pytest.raises(
            KeyError, match=r"Columns not found.*\['C', 'D'\]"
        )
        # Assert
        with ctx:
            merge_columns(df, "A", "C", "D")

    def test_single_column_merge_returns_column_with_string_values(self):
        # Arrange
        df = pd.DataFrame({"A": [1, 2, 3]})
        # Act
        result = merge_columns(df, "A", sep=" ")
        # Assert
        assert list(result["A"]) == ["1", "2", "3"]

    def test_empty_dataframe_merge_returns_empty_merged_column(self):
        # Arrange
        df = pd.DataFrame({"A": [], "B": []})
        # Act
        result = merge_columns(df, "A", "B", sep=" ")
        # Assert
        assert "A_B" in result.columns and len(result) == 0

    def test_large_number_of_columns_merge_concatenates_first_row(self):
        # Arrange
        data = {f"col{i}": list(range(3)) for i in range(10)}
        df = pd.DataFrame(data)
        cols = [f"col{i}" for i in range(10)]
        # Act
        result = merge_columns(df, *cols, sep=",")
        # Assert
        assert result["_".join(cols)].iloc[0] == ",".join(["0"] * 10)


class TestSpecialCharacters:
    """Test handling of special characters."""

    def test_columns_with_spaces_in_names_are_supported(self):
        # Arrange
        df = pd.DataFrame(
            {
                "First Name": ["John", "Jane"],
                "Last Name": ["Doe", "Smith"],
            }
        )
        # Act
        result = merge_columns(df, "First Name", "Last Name", sep=" ")
        # Assert
        assert result["First Name_Last Name"].iloc[0] == "John Doe"

    def test_pipe_separator_appears_between_values(self):
        # Arrange
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        # Act
        result = merge_columns(df, "A", "B", sep="||")
        # Assert
        assert result["A_B"].iloc[0] == "1||3"

    def test_tab_separator_appears_between_values(self):
        # Arrange
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        # Act
        result = merge_columns(df, "A", "B", sep="\t")
        # Assert
        assert result["A_B"].iloc[0] == "1\t3"

    def test_newline_separator_appears_between_values(self):
        # Arrange
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        # Act
        result = merge_columns(df, "A", "B", sep="\n")
        # Assert
        assert result["A_B"].iloc[0] == "1\n3"

    def test_unicode_string_content_concatenates_correctly(self):
        # Arrange
        df = pd.DataFrame(
            {
                "name": ["José", "François"],
                "city": ["São Paulo", "Montréal"],
            }
        )
        # Act
        result = merge_columns(df, "name", "city", sep=" - ")
        # Assert
        assert result["name_city"].iloc[0] == "José - São Paulo"


class TestDocstringExamples:
    """Test examples from the docstring."""

    def test_docstring_simple_separator_concatenates_values(self):
        # Arrange
        df = pd.DataFrame(
            {"A": [0, 5, 10], "B": [1, 6, 11], "C": [2, 7, 12]}
        )
        # Act
        result = merge_columns(df, "A", "B", sep=" ")
        # Assert
        assert result["A_B"].iloc[0] == "0 1"

    def test_docstring_column_labels_concatenate_with_labels(self):
        # Arrange
        df = pd.DataFrame(
            {"A": [0, 5, 10], "B": [1, 6, 11], "C": [2, 7, 12]}
        )
        # Act
        result = merge_columns(df, "A", "B", sep1="_", sep2="-")
        # Assert
        assert result["merged"].iloc[0] == "A-0_B-1"


class TestRealWorldScenarios:
    """Test real-world use cases."""

    def test_address_fields_concatenate_into_full_address_first(self):
        # Arrange
        df = pd.DataFrame(
            {
                "street": ["123 Main St", "456 Oak Ave"],
                "city": ["New York", "Los Angeles"],
                "state": ["NY", "CA"],
                "zip": ["10001", "90001"],
            }
        )
        # Act
        result = merge_columns(
            df,
            "street",
            "city",
            "state",
            "zip",
            sep=", ",
            name="full_address",
        )
        # Assert
        assert (
            result["full_address"].iloc[0]
            == "123 Main St, New York, NY, 10001"
        )

    def test_composite_key_creation_concatenates_with_underscore(self):
        # Arrange
        df = pd.DataFrame(
            {
                "year": [2023, 2023, 2024],
                "month": [1, 2, 1],
                "category": ["A", "B", "A"],
                "subcategory": ["X", "Y", "Z"],
            }
        )
        # Act
        result = merge_columns(
            df,
            "year",
            "month",
            "category",
            "subcategory",
            sep="_",
            name="composite_key",
        )
        # Assert
        assert result["composite_key"].iloc[0] == "2023_1_A_X"

    def test_log_message_creation_uses_custom_separators_per_field(self):
        # Arrange
        df = pd.DataFrame(
            {
                "timestamp": [
                    "2023-01-01 10:00:00",
                    "2023-01-01 10:01:00",
                ],
                "level": ["INFO", "ERROR"],
                "message": ["Process started", "Connection failed"],
            }
        )
        expected = (
            "timestamp: 2023-01-01 10:00:00 | level: INFO | "
            "message: Process started"
        )
        # Act
        result = merge_columns(
            df,
            "timestamp",
            "level",
            "message",
            sep1=" | ",
            sep2=": ",
            name="log_entry",
        )
        # Assert
        assert result["log_entry"].iloc[0] == expected


class TestPerformance:
    """Test performance-related scenarios."""

    def test_large_dataframe_merge_produces_expected_first_row(self):
        # Arrange
        n_rows = 10000
        df = pd.DataFrame(
            {
                "A": range(n_rows),
                "B": range(n_rows, 2 * n_rows),
                "C": [f"str_{i}" for i in range(n_rows)],
            }
        )
        # Act
        result = merge_columns(df, "A", "B", "C", sep="-")
        # Assert
        assert result["A_B_C"].iloc[0] == "0-10000-str_0"

    def test_merge_does_not_modify_original_dataframe(self):
        # Arrange
        df = pd.DataFrame({"X": [1, 2, 3], "Y": [4, 5, 6]})
        original = list(df.columns)
        # Act
        merge_columns(df, "X", "Y", sep=" ")
        # Assert
        assert list(df.columns) == original


class TestMergeColumnsEmpty:
    """Empty-DataFrame branch with a custom column name."""

    def test_empty_dataframe_with_custom_name_creates_named_column(self):
        # Arrange
        df = pd.DataFrame({"a": [], "b": []})
        # Act
        result = merge_columns(df, ["a", "b"], sep=None, name="custom")
        # Assert
        assert "custom" in result.columns


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__)])
