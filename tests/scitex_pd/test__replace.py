#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for scitex_pd.replace."""

import numpy as np
import pandas as pd
import pytest

from _helpers import frames_match, series_match
from scitex_pd import replace


class TestBasicReplacements:
    """Test basic replacement functionality."""

    def test_simple_string_replacement_updates_column_a(self):
        # Arrange
        df = pd.DataFrame(
            {"A": ["apple", "banana", "apple"], "B": ["orange", "apple", "grape"]}
        )
        # Act
        result = replace(df, "apple", "pear")
        # Assert
        assert result["A"].tolist() == ["pear", "banana", "pear"]

    def test_simple_string_replacement_updates_column_b(self):
        # Arrange
        df = pd.DataFrame(
            {"A": ["apple", "banana", "apple"], "B": ["orange", "apple", "grape"]}
        )
        # Act
        result = replace(df, "apple", "pear")
        # Assert
        assert result["B"].tolist() == ["orange", "pear", "grape"]

    def test_numeric_replacement_updates_int_column(self):
        # Arrange
        df = pd.DataFrame({"A": [1, 2, 3, 1], "B": [4, 1, 6, 7]})
        # Act
        result = replace(df, 1, 99)
        # Assert
        assert result["A"].tolist() == [99, 2, 3, 99]

    def test_dict_replacement_applies_each_mapping_to_column_a(self):
        # Arrange
        df = pd.DataFrame({"A": ["a", "b", "c"], "B": ["x", "y", "z"]})
        replace_dict = {"a": "alpha", "b": "beta", "x": "X", "z": "Z"}
        # Act
        result = replace(df, replace_dict)
        # Assert
        assert result["A"].tolist() == ["alpha", "beta", "c"]

    def test_dict_replacement_applies_each_mapping_to_column_b(self):
        # Arrange
        df = pd.DataFrame({"A": ["a", "b", "c"], "B": ["x", "y", "z"]})
        replace_dict = {"a": "alpha", "b": "beta", "x": "X", "z": "Z"}
        # Act
        result = replace(df, replace_dict)
        # Assert
        assert result["B"].tolist() == ["X", "y", "Z"]

    def test_specific_columns_replacement_skips_unselected_columns(self):
        # Arrange
        df = pd.DataFrame(
            {
                "A": ["test", "test", "other"],
                "B": ["test", "test", "test"],
                "C": ["test", "other", "test"],
            }
        )
        # Act
        result = replace(df, "test", "replaced", cols=["A", "C"])
        # Assert
        assert result["B"].tolist() == ["test", "test", "test"]

    def test_specific_columns_replacement_updates_target_columns(self):
        # Arrange
        df = pd.DataFrame(
            {
                "A": ["test", "test", "other"],
                "B": ["test", "test", "test"],
                "C": ["test", "other", "test"],
            }
        )
        # Act
        result = replace(df, "test", "replaced", cols=["A", "C"])
        # Assert
        assert result["A"].tolist() == ["replaced", "replaced", "other"]


class TestRegexReplacements:
    """Test regex-based replacements."""

    def test_simple_regex_strips_trailing_digit_suffix_from_a(self):
        # Arrange
        df = pd.DataFrame(
            {
                "A": ["abc-123", "def-456", "ghi-789"],
                "B": ["test-001", "test-002", "test-003"],
            }
        )
        # Act
        result = replace(df, r"-\d+", "", regex=True)
        # Assert
        assert result["A"].tolist() == ["abc", "def", "ghi"]

    def test_regex_dict_replacement_updates_email_addresses(self):
        # Arrange
        df = pd.DataFrame(
            {
                "A": ["email@domain.com", "user@test.org"],
                "B": ["phone: 123-456", "tel: 789-012"],
            }
        )
        replace_dict = {
            r"@.*\.com": "@company.com",
            r"@.*\.org": "@organization.org",
            r"\d{3}-\d{3}": "XXX-XXX",
        }
        # Act
        result = replace(df, replace_dict, regex=True)
        # Assert
        assert result["A"].tolist() == [
            "email@company.com",
            "user@organization.org",
        ]

    def test_regex_dict_replacement_updates_phone_format(self):
        # Arrange
        df = pd.DataFrame(
            {
                "A": ["email@domain.com", "user@test.org"],
                "B": ["phone: 123-456", "tel: 789-012"],
            }
        )
        replace_dict = {
            r"@.*\.com": "@company.com",
            r"@.*\.org": "@organization.org",
            r"\d{3}-\d{3}": "XXX-XXX",
        }
        # Act
        result = replace(df, replace_dict, regex=True)
        # Assert
        assert result["B"].tolist() == ["phone: XXX-XXX", "tel: XXX-XXX"]

    def test_regex_special_characters_strip_dollar_and_dot(self):
        # Arrange
        df = pd.DataFrame(
            {
                "A": ["$100.00", "$250.50", "$1000.99"],
                "B": ["#tag1", "#tag2", "#tag3"],
            }
        )
        # Act
        result = replace(df, r"\$|\.", "", regex=True)
        # Assert
        assert result["A"].tolist() == ["10000", "25050", "100099"]


class TestDataTypes:
    """Test replacements with different data types."""

    def test_mixed_type_replacement_updates_int_column_values(self):
        # Arrange
        df = pd.DataFrame(
            {
                "int": [1, 2, 3, 1],
                "float": [1.0, 2.5, 1.0, 3.5],
                "str": ["1", "2", "1", "3"],
                "bool": [True, False, True, False],
            }
        )
        # Act
        result = replace(df, 1, 99)
        # Assert
        assert result["int"].tolist() == [99, 2, 3, 99]

    def test_mixed_type_replacement_updates_float_column_values(self):
        # Arrange
        df = pd.DataFrame(
            {
                "int": [1, 2, 3, 1],
                "float": [1.0, 2.5, 1.0, 3.5],
            }
        )
        # Act
        result = replace(df, 1, 99)
        # Assert
        assert result["float"].tolist() == [99.0, 2.5, 99.0, 3.5]

    def test_mixed_type_replacement_leaves_string_column_unchanged(self):
        # Arrange
        df = pd.DataFrame(
            {
                "int": [1, 2, 3, 1],
                "str": ["1", "2", "1", "3"],
            }
        )
        # Act
        result = replace(df, 1, 99)
        # Assert
        assert result["str"].tolist() == ["1", "2", "1", "3"]

    def test_nan_replacement_updates_numeric_column_to_zero(self):
        # Arrange
        df = pd.DataFrame(
            {"A": [1, np.nan, 3, np.nan], "B": ["a", "b", np.nan, "d"]}
        )
        # Act
        result = replace(df, np.nan, 0)
        # Assert
        assert result["A"].tolist() == [1, 0, 3, 0]

    def test_none_replacement_updates_string_column_to_marker(self):
        # Arrange
        df = pd.DataFrame({"A": [1, None, 3], "B": ["a", None, "c"]})
        # Act
        result = replace(df, None, "missing")
        # Assert
        assert result["B"].tolist() == ["a", "missing", "c"]

    def test_datetime_replacement_swaps_target_dates_in_series(self):
        # Arrange
        df = pd.DataFrame(
            {
                "dates": pd.to_datetime(
                    ["2023-01-01", "2023-01-02", "2023-01-01"]
                ),
                "values": [1, 2, 3],
            }
        )
        old_date = pd.to_datetime("2023-01-01")
        new_date = pd.to_datetime("2023-01-15")
        expected = pd.Series(
            pd.to_datetime(["2023-01-15", "2023-01-02", "2023-01-15"]),
            name="dates",
        )
        # Act
        result = replace(df, old_date, new_date)
        # Assert
        assert series_match(result["dates"], expected)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_missing_new_value_with_string_old_raises_valueerror(self):
        # Arrange
        df = pd.DataFrame({"A": [1, 2, 3]})
        # Act
        ctx = pytest.raises(ValueError, match="new_value must be provided")
        # Assert
        with ctx:
            replace(df, "old")

    def test_empty_dataframe_returns_empty_dataframe(self):
        # Arrange
        df = pd.DataFrame()
        # Act
        result = replace(df, "old", "new")
        # Assert
        assert result.empty

    def test_nonexistent_column_in_cols_is_silently_skipped(self):
        # Arrange
        df = pd.DataFrame({"A": [1, 2, 3]})
        # Act
        result = replace(df, 1, 99, cols=["A", "B", "C"])
        # Assert
        assert result["A"].tolist() == [99, 2, 3]

    def test_no_matching_values_returns_input_dataframe_unchanged(self):
        # Arrange
        df = pd.DataFrame({"A": [1, 2, 3], "B": ["a", "b", "c"]})
        # Act
        result = replace(df, 99, 100)
        # Assert
        assert frames_match(result, df)

    def test_empty_replacement_dict_returns_input_dataframe_unchanged(self):
        # Arrange
        df = pd.DataFrame({"A": [1, 2, 3]})
        # Act
        result = replace(df, {})
        # Assert
        assert frames_match(result, df)


class TestComplexScenarios:
    """Test complex real-world scenarios."""

    def test_phone_normalization_regex_strips_non_digit_characters(self):
        # Arrange
        df = pd.DataFrame(
            {
                "name": ["John Doe", "Jane Smith", "Bob Johnson"],
                "phone": [
                    "123-456-7890",
                    "(555) 123-4567",
                    "999.888.7777",
                ],
                "email": [
                    "john@example.com",
                    "jane@test.org",
                    "bob@company.com",
                ],
            }
        )
        # Act
        result = replace(df, {r"[^\d]": ""}, regex=True, cols=["phone"])
        # Assert
        assert result["phone"].tolist() == [
            "1234567890",
            "5551234567",
            "9998887777",
        ]

    def test_categorical_mapping_replaces_size_aliases_with_canonical(self):
        # Arrange
        df = pd.DataFrame(
            {
                "size": ["S", "small", "M", "medium", "L", "large"],
                "color": [
                    "red",
                    "RED",
                    "Blue",
                    "BLUE",
                    "green",
                    "GREEN",
                ],
            }
        )
        size_map = {
            "S": "Small",
            "small": "Small",
            "M": "Medium",
            "medium": "Medium",
            "L": "Large",
            "large": "Large",
        }
        # Act
        result = replace(df, size_map, cols=["size"])
        # Assert
        assert result["size"].tolist() == [
            "Small",
            "Small",
            "Medium",
            "Medium",
            "Large",
            "Large",
        ]

    def test_chained_regex_replacements_compose_in_order(self):
        # Arrange
        df = pd.DataFrame(
            {"text": ["Hello World!", "Python Programming", "Data Science"]}
        )
        # Act
        result = replace(df, "Hello", "Hi", regex=True)
        result = replace(result, "Programming", "Coding", regex=True)
        result = replace(result, "!", ".", regex=True)
        # Assert
        assert result["text"].tolist() == [
            "Hi World.",
            "Python Coding",
            "Data Science",
        ]


class TestDocstringExample:
    """Test the example from the docstring."""

    def test_docstring_simple_regex_replacement_updates_first_row(self):
        # Arrange
        df = pd.DataFrame(
            {"A": ["abc-123", "def-456"], "B": ["ghi-789", "jkl-012"]}
        )
        # Act
        result = replace(df, "abc", "xyz", regex=True)
        # Assert
        assert result["A"].iloc[0] == "xyz-123"

    def test_docstring_dict_replacement_targets_only_supplied_columns(self):
        # Arrange
        df = pd.DataFrame(
            {"A": ["abc-123", "def-456"], "B": ["ghi-789", "jkl-012"]}
        )
        replace_dict = {"-": "_", "1": "one"}
        # Act
        result = replace(df, replace_dict, regex=True, cols=["A"])
        # Assert
        assert result["A"].iloc[0] == "abc_one23"

    def test_docstring_dict_replacement_leaves_other_columns_unchanged(self):
        # Arrange
        df = pd.DataFrame(
            {"A": ["abc-123", "def-456"], "B": ["ghi-789", "jkl-012"]}
        )
        replace_dict = {"-": "_", "1": "one"}
        # Act
        result = replace(df, replace_dict, regex=True, cols=["A"])
        # Assert
        assert result["B"].iloc[0] == "ghi-789"


class TestPreservation:
    """Test that original DataFrame is not modified."""

    def test_original_dataframe_column_a_unchanged_after_replace(self):
        # Arrange
        df = pd.DataFrame({"A": [1, 2, 3], "B": ["a", "b", "c"]})
        original_a = df["A"].copy()
        # Act
        replace(df, 1, 99)
        # Assert
        assert series_match(df["A"], original_a)

    def test_index_is_preserved_after_replacement(self):
        # Arrange
        df = pd.DataFrame({"A": [1, 2, 3]}, index=["x", "y", "z"])
        # Act
        result = replace(df, 2, 99)
        # Assert
        assert list(result.index) == ["x", "y", "z"]

    def test_column_order_is_preserved_after_replacement(self):
        # Arrange
        df = pd.DataFrame({"Z": [1, 2], "A": [3, 4], "M": [5, 6]})
        # Act
        result = replace(df, 1, 99)
        # Assert
        assert list(result.columns) == ["Z", "A", "M"]


class TestLargeDatasets:
    """Test with larger datasets."""

    def test_large_dataframe_replaces_mapped_string_values(self):
        # Arrange
        np.random.seed(0)
        n = 10000
        df = pd.DataFrame(
            {
                "A": np.random.choice(["a", "b", "c"], n),
                "B": np.random.randint(0, 10, n),
                "C": np.random.choice(["x", "y", "z"], n),
            }
        )
        # Act
        result = replace(df, {"a": "alpha", "b": "beta", 5: 555})
        original_a = set(df["A"].unique())
        result_a = set(result["A"].unique())
        # Assert
        assert "a" not in result_a and "alpha" in result_a


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__)])
