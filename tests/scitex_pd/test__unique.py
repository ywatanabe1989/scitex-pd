#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for scitex_pd.unique / uq."""
from __future__ import annotations

import numpy as np
import pandas as pd

from scitex_pd import unique, uq


class TestUnique:
    def test_returns_dataframe(self):
        result = unique([1, 2, 2, 3, 3, 3])
        assert isinstance(result, pd.DataFrame)

    def test_columns_default(self):
        result = unique([1, 2, 2, 3, 3, 3])
        assert "Unique Elements" in result.columns
        assert "Counts" in result.columns

    def test_values_and_counts(self):
        result = unique([1, 2, 2, 3, 3, 3])
        assert result["Unique Elements"].tolist() == [1, 2, 3]
        # Counts are formatted with thousands separators (str type).
        assert result["Counts"].tolist() == ["1", "2", "3"]

    def test_thousand_separator_format(self):
        big = np.concatenate([np.full(2500, 0), np.full(1500, 1)])
        result = unique(big)
        assert result["Counts"].tolist() == ["2,500", "1,500"]

    def test_axis_kwarg(self):
        arr = np.array([[1, 2], [1, 2], [3, 4]])
        result = unique(arr, axis=0)
        # Two unique rows: (1,2) appears twice, (3,4) once.
        assert "Unique Elements Axis 0" in result.columns
        assert "Counts" in result.columns


class TestUq:
    def test_alias_for_unique(self):
        a = unique([1, 2, 3])
        b = uq([1, 2, 3])
        pd.testing.assert_frame_equal(a, b)
