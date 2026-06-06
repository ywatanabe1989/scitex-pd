#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""unique / uq — frequency-table helper ported from scitex-gen misc.py.

The upstream module defined ``unique`` twice — the second definition
shadowed the first, so this port follows the second (winning)
definition which uses "Unique Elements" / "Counts" column names.

This is distinct from ``scitex_pd.get_unique`` (which returns a
*single* scalar if the column is mono-valued). The ``unique`` helper
here returns a *DataFrame* tabulating every value and its count.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def unique(data, axis: Optional[int] = None) -> pd.DataFrame:
    """Tabulate unique values in ``data`` with their counts.

    Parameters
    ----------
    data : array-like
        Input data — anything :func:`numpy.unique` accepts.
    axis : int, optional
        Axis along which to find unique elements. ``None`` (default)
        flattens the input.

    Returns
    -------
    pandas.DataFrame
        With columns ``Unique Elements`` and ``Counts`` (or
        ``Unique Elements Axis i`` for multi-axis inputs). The
        ``Counts`` column is formatted with thousands separators.

    Example
    -------
    >>> result = unique([1, 2, 2, 3, 3, 3])
    >>> result["Unique Elements"].tolist()
    [1, 2, 3]
    >>> result["Counts"].tolist()
    ['1', '2', '3']
    """
    if axis is None:
        uqs, counts = np.unique(data, return_counts=True)
        df = pd.DataFrame({"Unique Elements": uqs, "Counts": counts})
    else:
        uqs, counts = np.unique(data, axis=axis, return_counts=True)
        df = pd.DataFrame(
            uqs,
            columns=[f"Unique Elements Axis {i}" for i in range(uqs.shape[1])],
        )
        df["Counts"] = counts

    df["Counts"] = df["Counts"].apply(lambda x: f"{x:,}")
    return df


def uq(*args, **kwargs) -> pd.DataFrame:
    """Alias for :func:`unique`. Provided for backward-compatibility.

    See Also
    --------
    unique : The full-name function.
    """
    return unique(*args, **kwargs)
