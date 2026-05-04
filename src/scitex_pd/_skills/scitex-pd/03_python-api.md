---
description: |
  [TOPIC] Python API
  [DETAILS] All public callables exported by scitex_pd — coercion, reshaping, column ops, finders, numerics, warnings.
tags: [scitex-pd-python-api]
---

# Python API

```python
import scitex_pd as spd
```

## Coercion + reshape

| Callable | Purpose |
|---|---|
| `force_df(x)` | Series/dict/list/ndarray → DataFrame |
| `melt_cols(df, id_vars, value_pattern)` | Wide → long via column-name regex |
| `from_xyz(df)` | Long xyz → wide pivot |
| `to_xy(df)` | Coord DataFrame → x/y columns |
| `to_xyz(df)` | Wide → long xyz |

## Column-by-name ops

| Callable | Purpose |
|---|---|
| `mv(df, col, before=...)` | Reorder a single column |
| `mv_to_first(df, col)` | Move column to leftmost position |
| `mv_to_last(df, col)` | Move column to rightmost position |
| `merge_cols(df, cols, into=..., sep=...)` / `merge_columns` | Concat columns into one |
| `replace(df, mapping, columns=...)` | Targeted value replacement |
| `slice(df, where=...)` | Boolean slice via dict spec |
| `sort(df, by=..., desc=...)` | Sort with directional flag |

## Statistical finders

| Callable | Purpose |
|---|---|
| `find_pval(df)` | Locate the p-value column by convention |
| `find_indi(df)` | Locate indicator/dummy columns |

## Convenience numerics

| Callable | Purpose |
|---|---|
| `to_numeric(df, errors=...)` | DataFrame-wide numeric coercion |
| `round(df, decimals=...)` | DataFrame-wide rounding |
| `get_unique(df, columns=...)` | Per-column unique values |

## Warning suppression

| Callable | Purpose |
|---|---|
| `ignore_setting_with_copy_warning()` | Context manager — suppress pandas chained-assignment warning |
| `ignore_SettingWithCopyWarning` | PEP-8-violating alias preserved for back-compat |

## See also

- `scitex-stats` — consumes `find_pval` outputs
- `scitex-plt` — pairs with `to_xy` / `from_xyz`
