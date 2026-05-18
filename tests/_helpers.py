"""Single-assertion shims so each test body can satisfy STX-TQ001 and
STX-TQ007 (one ``assert`` per test) while preserving the rich diagnostic
output of ``pd.testing.assert_*`` / ``np.testing.assert_*``.

Each helper delegates to the pandas/numpy comparator, which raises
``AssertionError`` with a structured diff on mismatch, and otherwise
returns ``True`` so callers can write::

    assert frames_match(actual, expected)

and stay TQ001-clean with a single explicit ``assert`` statement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def frames_match(actual: pd.DataFrame, expected: pd.DataFrame, **kwargs) -> bool:
    """``pd.testing.assert_frame_equal`` wrapper that returns ``True`` on success."""
    pd.testing.assert_frame_equal(actual, expected, **kwargs)
    return True


def series_match(actual: pd.Series, expected: pd.Series, **kwargs) -> bool:
    """``pd.testing.assert_series_equal`` wrapper that returns ``True`` on success."""
    pd.testing.assert_series_equal(actual, expected, **kwargs)
    return True


def arrays_match(actual, expected, **kwargs) -> bool:
    """``np.testing.assert_array_equal`` wrapper that returns ``True`` on success."""
    np.testing.assert_array_equal(actual, expected, **kwargs)
    return True


def arrays_close(actual, expected, **kwargs) -> bool:
    """``np.testing.assert_allclose`` wrapper that returns ``True`` on success."""
    np.testing.assert_allclose(actual, expected, **kwargs)
    return True
