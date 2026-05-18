#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for scitex_pd.ignore_setting_with_copy_warning."""

import warnings

import numpy as np
import pandas as pd
import pytest


# pandas >=2.2 removed SettingWithCopyWarning entirely (Copy-on-Write is now
# the default). The tests in this module exercise a context manager that
# silences that warning class; if pandas no longer exposes it, the whole
# behaviour under test is moot, so skip the module rather than fail.
def _settingwithcopywarning_available() -> bool:
    try:
        from pandas.errors import SettingWithCopyWarning  # noqa: F401

        return True
    except ImportError:
        pass
    try:
        from pandas.core.common import SettingWithCopyWarning  # noqa: F401

        return True
    except ImportError:
        return False


# The warning-suppression *behaviour* only fires when pandas still exposes
# SettingWithCopyWarning (pandas <2.2). On newer pandas the class is gone
# and the contextmanager degrades to a pure no-op. Tests that assert real
# suppression are class-marked with this skip; unconditional tests live in
# ``TestIgnoreSettingWithCopyWarningUniversal`` and run on every pandas.
_skip_without_swcw = pytest.mark.skipif(
    not _settingwithcopywarning_available(),
    reason="pandas >=2.2 removed SettingWithCopyWarning; nothing to suppress.",
)


def _count_setting_with_copy_warnings(records) -> int:
    return sum(1 for w in records if "SettingWithCopy" in str(w.category))


@_skip_without_swcw
class TestIgnoreSettingWithCopyWarningBasic:
    """Test basic functionality of ignore_setting_with_copy_warning."""

    def test_context_manager_suppresses_warning_on_slice_assignment(self):
        # Arrange
        from scitex_pd import ignore_setting_with_copy_warning
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        df_view = df[df["A"] > 1]
        # Act
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            with ignore_setting_with_copy_warning():
                df_view["B"] = 99
            suppressed = _count_setting_with_copy_warnings(records)
        # Assert
        assert suppressed == 0

    def test_context_manager_suppresses_warning_on_loc_assignment(self):
        # Arrange
        from scitex_pd import ignore_setting_with_copy_warning
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        subset = df[["A"]]
        # Act
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            with ignore_setting_with_copy_warning():
                subset.loc[:, "A"] = 100
            suppressed = _count_setting_with_copy_warnings(records)
        # Assert
        assert suppressed == 0


@_skip_without_swcw
class TestBackwardCompatibility:
    """Test backward compatibility with old function name."""

    def test_camelcase_alias_suppresses_setting_with_copy_warning(self):
        # Arrange
        from scitex_pd import ignore_SettingWithCopyWarning
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        df_view = df[df["A"] > 1]
        # Act
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            with ignore_SettingWithCopyWarning():
                df_view["B"] = 99
            suppressed = _count_setting_with_copy_warnings(records)
        # Assert
        assert suppressed == 0

    def test_snake_case_and_camel_case_names_refer_to_same_callable(self):
        # Arrange
        from scitex_pd import (
            ignore_setting_with_copy_warning,
            ignore_SettingWithCopyWarning,
        )
        # Act
        same = ignore_setting_with_copy_warning is ignore_SettingWithCopyWarning
        # Assert
        assert same


@_skip_without_swcw
class TestComplexScenarios:
    """Test complex DataFrame manipulation scenarios."""

    def test_chained_indexing_under_context_suppresses_warning(self):
        # Arrange
        from scitex_pd import ignore_setting_with_copy_warning
        df = pd.DataFrame(
            {"A": [1, 2, 3, 4], "B": [5, 6, 7, 8], "C": ["x", "y", "z", "w"]}
        )
        # Act
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            with ignore_setting_with_copy_warning():
                df[df["A"] > 2]["B"] = 999
            suppressed = _count_setting_with_copy_warnings(records)
        # Assert
        assert suppressed == 0

    def test_multiple_view_writes_in_one_block_suppress_warning(self):
        # Arrange
        from scitex_pd import ignore_setting_with_copy_warning
        df = pd.DataFrame({"A": range(10), "B": range(10, 20), "C": range(20, 30)})
        view1 = df[df["A"] < 5]
        view2 = df[["B", "C"]]
        # Act
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            with ignore_setting_with_copy_warning():
                view1["B"] = -1
                view2["C"] = -2
                view1.loc[:, "C"] = -3
            suppressed = _count_setting_with_copy_warnings(records)
        # Assert
        assert suppressed == 0

    def test_nested_dataframe_loop_in_context_suppresses_warning(self):
        # Arrange
        from scitex_pd import ignore_setting_with_copy_warning
        df = pd.DataFrame({"group": ["A", "A", "B", "B"], "value": [1, 2, 3, 4]})
        # Act
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            with ignore_setting_with_copy_warning():
                for group in df["group"].unique():
                    group_df = df[df["group"] == group]
                    group_df["value"] = group_df["value"] * 10
            suppressed = _count_setting_with_copy_warnings(records)
        # Assert
        assert suppressed == 0


@_skip_without_swcw
class TestWarningRestoration:
    """Test that warning settings are properly restored."""

    def test_warnings_work_normally_after_context_exits(self):
        # Arrange
        from scitex_pd import ignore_setting_with_copy_warning
        with ignore_setting_with_copy_warning():
            pass
        # Act
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            warnings.warn("Test warning", UserWarning)
        # Assert
        assert len(records) == 1

    def test_exception_inside_context_propagates_to_caller(self):
        # Arrange
        from scitex_pd import ignore_setting_with_copy_warning
        # Act
        ctx = pytest.raises(ValueError)
        # Assert
        with ctx:
            with ignore_setting_with_copy_warning():
                raise ValueError("Test exception")


@_skip_without_swcw
class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_context_does_not_raise_any_exception(self):
        # Arrange
        from scitex_pd import ignore_setting_with_copy_warning
        sentinel = []
        # Act
        with ignore_setting_with_copy_warning():
            sentinel.append("ran")
        # Assert
        assert sentinel == ["ran"]

    def test_other_warning_categories_still_propagate_inside_context(self):
        # Arrange
        from scitex_pd import ignore_setting_with_copy_warning
        # Act
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            with ignore_setting_with_copy_warning():
                arr = np.array([1, 2, 3])
                arr[0] = 999
                warnings.warn("Test warning", UserWarning)
            user_warning_count = sum(
                1 for r in records if issubclass(r.category, UserWarning)
            )
        # Assert
        assert user_warning_count == 1

    def test_nested_context_managers_suppress_warnings_in_both_scopes(self):
        # Arrange
        from scitex_pd import ignore_setting_with_copy_warning
        df1 = pd.DataFrame({"A": [1, 2, 3]})
        df2 = pd.DataFrame({"B": [4, 5, 6]})
        view1 = df1[df1["A"] > 1]
        view2 = df2[df2["B"] < 6]
        # Act
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            with ignore_setting_with_copy_warning():
                view1["A"] = 10
                with ignore_setting_with_copy_warning():
                    view2["B"] = 20
            suppressed = _count_setting_with_copy_warnings(records)
        # Assert
        assert suppressed == 0


@_skip_without_swcw
class TestRealWorldUsage:
    """Test real-world usage patterns."""

    def test_typical_data_cleaning_workflow_suppresses_warning(self):
        # Arrange
        from scitex_pd import ignore_setting_with_copy_warning
        df = pd.DataFrame(
            {
                "id": range(100),
                "value": np.random.randn(100),
                "category": np.random.choice(["A", "B", "C"], 100),
            }
        )
        df_filtered = df[df["value"] > 0]
        # Act
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            with ignore_setting_with_copy_warning():
                df_filtered["value"] = df_filtered["value"].round(2)
                df_filtered["processed"] = True
                df_filtered.loc[df_filtered["category"] == "A", "special"] = "yes"
            suppressed = _count_setting_with_copy_warnings(records)
        # Assert
        assert suppressed == 0

    def test_iterative_view_updates_inside_context_suppress_warning(self):
        # Arrange
        from scitex_pd import ignore_setting_with_copy_warning
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=30),
                "value": np.random.randn(30),
            }
        )
        january = df[df["date"].dt.month == 1]
        # Act
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            with ignore_setting_with_copy_warning():
                for i in range(len(january)):
                    if january.iloc[i]["value"] < 0:
                        january.iloc[i, january.columns.get_loc("value")] = 0
            suppressed = _count_setting_with_copy_warnings(records)
        # Assert
        assert suppressed == 0


@_skip_without_swcw
class TestDocstringExample:
    """Test the example from the docstring."""

    def test_docstring_example_suppresses_setting_with_copy_warning(self):
        # Arrange
        from scitex_pd import ignore_setting_with_copy_warning
        df = pd.DataFrame({"column": [1, 2, 3], "other": [4, 5, 6]})
        df_subset = df[df["column"] > 1]
        new_values = [10, 20]
        # Act
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            with ignore_setting_with_copy_warning():
                df_subset["column"] = new_values
            suppressed = _count_setting_with_copy_warnings(records)
        # Assert
        assert suppressed == 0


class TestIgnoreSettingWithCopyWarningUniversal:
    """Run on every pandas version. Covers the no-op path (pandas >=2.2)
    AND, via a temporary attribute injection, the simplefilter path that
    is otherwise unreachable on modern pandas."""

    def test_context_manager_yields_cleanly_without_exception(self):
        # Arrange
        from scitex_pd import ignore_setting_with_copy_warning
        # Act
        with ignore_setting_with_copy_warning():
            value = 1 + 1
        # Assert
        assert value == 2

    def test_camel_case_alias_resolves_to_same_callable_object(self):
        # Arrange
        from scitex_pd import (
            ignore_setting_with_copy_warning,
            ignore_SettingWithCopyWarning,
        )
        # Act
        same = ignore_SettingWithCopyWarning is ignore_setting_with_copy_warning
        # Assert
        assert same

    def test_simplefilter_path_suppresses_injected_warning_class(self):
        # Arrange
        import pandas.errors
        from scitex_pd import ignore_setting_with_copy_warning

        had_attr = hasattr(pandas.errors, "SettingWithCopyWarning")
        original = getattr(pandas.errors, "SettingWithCopyWarning", None)

        class _FakeSWCW(Warning):
            pass

        pandas.errors.SettingWithCopyWarning = _FakeSWCW
        # Act
        try:
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                with ignore_setting_with_copy_warning():
                    warnings.warn("ignored", _FakeSWCW)
            leaked = any(isinstance(rec.message, _FakeSWCW) for rec in captured)
        finally:
            if had_attr:
                pandas.errors.SettingWithCopyWarning = original
            else:
                delattr(pandas.errors, "SettingWithCopyWarning")
        # Assert
        assert not leaked


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__)])
