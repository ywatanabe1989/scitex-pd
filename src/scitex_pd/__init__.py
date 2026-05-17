#!/usr/bin/env python3
"""scitex-pd — pandas helpers extracted from the SciTeX ecosystem.

Functionalities
---------------
- `force_df(x)` — coerce dict / Series / list / scalar / ndarray
  into a `pandas.DataFrame` with sensible defaults.
- `from_xyz(df, x, y, z)` / `to_xy(df)` / `to_xyz(df)` — long ↔ wide
  pivots; `to_numeric(df)` — column-wise numeric coercion.
- `find_pval(df)` / `_find_pval_col` — locate p-value columns by name.
- `find_indi(df, mask)` / `get_unique(df, col)` — boolean-mask and
  unique-values inspection helpers.
- `merge_columns` / `merge_cols`, `melt_cols`, `mv` /
  `mv_to_first` / `mv_to_last` — column combine / reshape / reorder.
- `replace(df, mapping)`, `round(df, ndigits)`, `slice(df, ...)`,
  `sort(df, ...)` — uniform DataFrame-in / DataFrame-out transforms.
- `ignore_setting_with_copy_warning()` — context-manager for the
  pandas SettingWithCopyWarning.

IO
--
- Reads: `pandas.DataFrame`, `pandas.Series`, `numpy.ndarray`, dict,
  list, scalar inputs.
- Writes: nothing — all functions return new pandas objects; original
  inputs are not mutated.

Dependencies
------------
- Hard: `pandas`, `numpy`, `scitex-types` (for `is_listed_X`).

Standalone import::

    import scitex_pd as pd_
    df = pd_.force_df(data)
    pvals = pd_.find_pval(df)

The umbrella `scitex.pd` import path is preserved via a
`sys.modules`-alias bridge in `scitex-python`.
"""

from __future__ import annotations

try:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _v

    try:
        __version__ = _v("scitex-pd")
    except PackageNotFoundError:
        __version__ = "0.0.0+local"
    del _v, PackageNotFoundError
except ImportError:  # pragma: no cover — only on ancient Pythons
    __version__ = "0.0.0+local"
from ._convert._from_xyz import from_xyz
from ._convert._to_numeric import to_numeric
from ._convert._to_xy import to_xy
from ._convert._to_xyz import to_xyz
from ._find_indi import find_indi
from ._find_pval import _find_pval_col, find_pval
from ._force_df import force_df
from ._get_unique import get_unique
from ._ignore_SettingWithCopyWarning import (
    ignore_setting_with_copy_warning,
    ignore_SettingWithCopyWarning,
)
from ._melt_cols import melt_cols
from ._merge_columns import merge_cols, merge_columns
from ._mv import mv, mv_to_first, mv_to_last
from ._replace import replace
from ._round import round
from ._slice import slice
from ._sort import sort

__all__ = [
    "__version__",
    "find_indi",
    "find_pval",
    "force_df",
    "from_xyz",
    "get_unique",
    "ignore_SettingWithCopyWarning",
    "ignore_setting_with_copy_warning",
    "melt_cols",
    "merge_cols",
    "merge_columns",
    "mv",
    "mv_to_first",
    "mv_to_last",
    "replace",
    "round",
    "slice",
    "sort",
    "to_numeric",
    "to_xy",
    "to_xyz",
]
