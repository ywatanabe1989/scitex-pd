#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for scitex_pd.find_pval."""

import numpy as np
import pandas as pd
import pytest

from scitex_pd import find_pval


class TestFindPvalDataFrame:
    """Test find_pval with DataFrame inputs."""

    def test_single_pval_column_returns_column_name(self):
        # Arrange
        df = pd.DataFrame({"p_value": [0.05, 0.01], "other": [1, 2]})
        # Act
        result = find_pval(df, multiple=False)
        # Assert
        assert result == "p_value"

    def test_multiple_pval_columns_returns_list_type(self):
        # Arrange
        df = pd.DataFrame(
            {
                "p_value": [0.05, 0.01],
                "pval": [0.1, 0.001],
                "p-val": [0.2, 0.02],
                "other": [1, 2],
            }
        )
        # Act
        result = find_pval(df, multiple=True)
        # Assert
        assert isinstance(result, list)

    def test_multiple_pval_columns_returns_each_match(self):
        # Arrange
        df = pd.DataFrame(
            {
                "p_value": [0.05, 0.01],
                "pval": [0.1, 0.001],
                "p-val": [0.2, 0.02],
                "other": [1, 2],
            }
        )
        # Act
        result = find_pval(df, multiple=True)
        # Assert
        assert set(result) == {"p_value", "pval", "p-val"}

    def test_pvalue_variations_match_every_listed_alias(self):
        # Arrange
        df = pd.DataFrame(
            {
                "pval": [0.1],
                "p_val": [0.2],
                "p-val": [0.3],
                "pvalue": [0.4],
                "p_value": [0.5],
                "p-value": [0.6],
                "Pval": [0.7],
                "PVALUE": [0.8],
                "P_Value": [0.9],
            }
        )
        # Act
        result = find_pval(df, multiple=True)
        # Assert
        assert all(col in result for col in df.columns)

    def test_no_pval_columns_with_multiple_false_returns_none(self):
        # Arrange
        df = pd.DataFrame({"alpha": [0.05], "beta": [0.1], "gamma": [1]})
        # Act
        result = find_pval(df, multiple=False)
        # Assert
        assert result is None

    def test_no_pval_columns_with_multiple_true_returns_empty_list(self):
        # Arrange
        df = pd.DataFrame({"alpha": [0.05], "beta": [0.1], "gamma": [1]})
        # Act
        result = find_pval(df, multiple=True)
        # Assert
        assert result == []

    def test_pval_stars_columns_are_excluded_from_matches(self):
        # Arrange
        df = pd.DataFrame(
            {
                "p_value": [0.05],
                "pval_stars": ["*"],
                "p_value_stars": ["**"],
                "pvalstars": ["***"],
            }
        )
        # Act
        result = find_pval(df, multiple=True)
        # Assert
        assert result == ["p_value"]

    def test_empty_dataframe_returns_empty_list_for_multiple(self):
        # Arrange
        df = pd.DataFrame()
        # Act
        result = find_pval(df, multiple=True)
        # Assert
        assert result == []


class TestFindPvalDict:
    """Test find_pval with dictionary inputs."""

    def test_dict_single_match_returns_key_name(self):
        # Arrange
        data = {"p_value": 0.05, "coefficient": 1.2, "se": 0.1}
        # Act
        result = find_pval(data, multiple=False)
        # Assert
        assert result == "p_value"

    def test_dict_multiple_matches_returns_all_pval_keys(self):
        # Arrange
        data = {"p_value": 0.05, "pval": 0.01, "p-val": 0.02, "coefficient": 1.2}
        # Act
        result = find_pval(data, multiple=True)
        # Assert
        assert set(result) == {"p_value", "pval", "p-val"}

    def test_dict_no_matches_returns_none_for_single(self):
        # Arrange
        data = {"alpha": 0.05, "beta": 0.1, "gamma": 1}
        # Act
        result = find_pval(data, multiple=False)
        # Assert
        assert result is None

    def test_nested_dict_only_top_level_keys_are_inspected(self):
        # Arrange
        data = {"results": {"p_value": 0.05}, "p_val": 0.01}
        # Act
        result = find_pval(data, multiple=True)
        # Assert
        assert result == ["p_val"]


class TestFindPvalList:
    """Test find_pval with list inputs."""

    def test_list_of_dicts_uses_first_record_for_key_lookup(self):
        # Arrange
        data = [
            {"p_value": 0.05, "coef": 1.2},
            {"p_value": 0.01, "coef": 2.3},
            {"p_value": 0.001, "coef": 3.4},
        ]
        # Act
        result = find_pval(data, multiple=False)
        # Assert
        assert result == "p_value"

    def test_list_of_dicts_returns_each_pval_alias_in_first_record(self):
        # Arrange
        data = [
            {"p_value": 0.05, "pval": 0.06, "coef": 1.2},
            {"p_value": 0.01, "pval": 0.02, "coef": 2.3},
        ]
        # Act
        result = find_pval(data, multiple=True)
        # Assert
        assert set(result) == {"p_value", "pval"}

    def test_empty_list_returns_empty_list_for_multiple(self):
        # Arrange
        data = []
        # Act
        result = find_pval(data, multiple=True)
        # Assert
        assert result == []

    def test_list_of_non_dicts_returns_none_for_single(self):
        # Arrange
        data = [1, 2, 3, 4]
        # Act
        result = find_pval(data, multiple=False)
        # Assert
        assert result is None


class TestFindPvalNumPy:
    """Test find_pval with numpy array inputs."""

    def test_numpy_array_of_dicts_uses_first_dict_keys(self):
        # Arrange
        data = np.array(
            [{"p_value": 0.05, "stat": 2.1}, {"p_value": 0.01, "stat": 3.2}]
        )
        # Act
        result = find_pval(data, multiple=False)
        # Assert
        assert result == "p_value"

    def test_plain_numpy_array_returns_none_for_single(self):
        # Arrange
        data = np.array([1, 2, 3])
        # Act
        result = find_pval(data, multiple=False)
        # Assert
        assert result is None

    def test_empty_numpy_array_returns_empty_list_for_multiple(self):
        # Arrange
        data = np.array([])
        # Act
        result = find_pval(data, multiple=True)
        # Assert
        assert result == []


class TestFindPvalEdgeCases:
    """Test edge cases and error handling."""

    def test_case_insensitive_matching_picks_up_uppercase_aliases(self):
        # Arrange
        df = pd.DataFrame(
            {
                "P_VALUE": [0.05],
                "Pval": [0.01],
                "P-Val": [0.02],
                "PVALUE": [0.03],
            }
        )
        # Act
        result = find_pval(df, multiple=True)
        # Assert
        assert len(result) == 4

    def test_numeric_column_names_do_not_break_pval_lookup(self):
        # Arrange
        df = pd.DataFrame({0: [1, 2], 1: [3, 4], "p_value": [0.05, 0.01]})
        # Act
        result = find_pval(df, multiple=False)
        # Assert
        assert result == "p_value"

    def test_special_characters_only_match_when_pattern_allows(self):
        # Arrange
        df = pd.DataFrame(
            {
                "p.value": [0.05],
                "p$val": [0.01],
                "p_value!": [0.02],
                "normal": [1],
            }
        )
        # Act
        result = find_pval(df, multiple=True)
        # Assert
        assert result == ["p_value!"]

    def test_invalid_input_type_raises_valueerror_with_message(self):
        # Arrange
        bad = "invalid_input"
        # Act
        ctx = pytest.raises(ValueError, match="Input must be a pandas DataFrame")
        # Assert
        with ctx:
            find_pval(bad)

    def test_partial_match_excludes_unrelated_columns(self):
        # Arrange
        df = pd.DataFrame(
            {
                "pval_test": [0.05],
                "test_pvalue": [0.01],
                "my_p_value_column": [0.02],
                "not_related": [1],
            }
        )
        # Act
        result = find_pval(df, multiple=True)
        # Assert
        assert "not_related" not in result


class TestFindPvalDocumentation:
    """Test examples from documentation."""

    def test_docstring_example_returns_pval_aliases_for_multiple(self):
        # Arrange
        df = pd.DataFrame(
            {"p_value": [0.05, 0.01], "pval": [0.1, 0.001], "other": [1, 2]}
        )
        # Act
        result = find_pval(df)
        # Assert
        assert set(result) == {"p_value", "pval"}

    def test_docstring_example_returns_first_alias_for_single(self):
        # Arrange
        df = pd.DataFrame(
            {"p_value": [0.05, 0.01], "pval": [0.1, 0.001], "other": [1, 2]}
        )
        # Act
        result = find_pval(df, multiple=False)
        # Assert
        assert result == "p_value"

    def test_private_alias_returns_same_result_as_public_helper(self):
        # Arrange
        from scitex_pd import _find_pval_col

        df = pd.DataFrame({"p_value": [0.05], "data": [10]})
        # Act
        result = _find_pval_col(df, multiple=False)
        # Assert
        assert result == "p_value"


class TestFindPvalIntegration:
    """Integration tests with real-world scenarios."""

    def test_typical_statistical_results_dataframe_finds_p_value(self):
        # Arrange
        df = pd.DataFrame(
            {
                "variable": ["age", "gender", "treatment"],
                "coefficient": [0.5, -0.3, 1.2],
                "std_error": [0.1, 0.2, 0.3],
                "t_statistic": [5.0, -1.5, 4.0],
                "p_value": [0.001, 0.134, 0.002],
                "confidence_lower": [0.3, -0.7, 0.6],
                "confidence_upper": [0.7, 0.1, 1.8],
            }
        )
        # Act
        result = find_pval(df, multiple=False)
        # Assert
        assert result == "p_value"

    def test_multiple_test_results_list_finds_pval_key(self):
        # Arrange
        results = [
            {"test": "t-test", "statistic": 2.5, "pval": 0.012},
            {"test": "chi-square", "statistic": 5.3, "pval": 0.021},
            {"test": "anova", "statistic": 3.8, "pval": 0.052},
        ]
        # Act
        result = find_pval(results)
        # Assert
        assert result == ["pval"]


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__)])
