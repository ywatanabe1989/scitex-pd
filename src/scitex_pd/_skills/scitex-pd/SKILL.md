---
name: scitex-pd
description: pandas helpers extracted from the SciTeX ecosystem. `force_df(x)` coerces anything (Series, dict, list-of-dicts, ndarray) into a `DataFrame` with sensible column names. `melt_cols(df, id_vars, ...)` is `pd.melt` with column-name patterns instead of explicit lists. `find_pval(df)` / `find_indi(df)` locate p-value and indicator columns by convention. `to_xy`/`to_xyz`/`from_xyz` reshape between long-form coordinate frames and wide. `mv`, `mv_to_first`, `mv_to_last` reorder columns by name without a verbose `df[[...]]` assignment. `get_unique`, `merge_cols`, `replace`, `round`, `slice`, `sort`, `to_numeric` are short-form helpers for the most common dataframe-massaging steps. Drop-in replacement for ten-line `if isinstance(x, ...): elif ...: ...` constructors and bespoke `df.columns.tolist()[:i] + ['x'] + df.columns.tolist()[i+1:]` reorderings.
primary_interface: python
interfaces:
  python: 3
  cli: 0
  mcp: 0
  skills: 2
  hook: 0
  http: 0
canonical-location: scitex-pd/src/scitex_pd/_skills/scitex-pd/SKILL.md
tags: [scitex-pd, scitex-package, pandas, dataframe, helpers]
---

> **Interfaces:** Python ⭐⭐⭐ (primary) · CLI — · MCP — · Skills ⭐⭐ · Hook — · HTTP —

# scitex-pd

Conventional pandas helpers — replaces the boilerplate that everyone
re-writes when massaging dataframes for analysis.

## Coercion + reshape

```python
import scitex_pd as spd

df = spd.force_df(any_input)            # Series/dict/list/ndarray → DataFrame
long = spd.melt_cols(df, id_vars=["subject"], value_pattern=r"trial_\d+")
wide = spd.from_xyz(long_xyz_df)
xy = spd.to_xy(coord_df)
```

## Column-by-name ops

```python
spd.mv(df, "score", before="label")     # reorder
spd.mv_to_first(df, "id")
spd.mv_to_last(df, "notes")
spd.merge_cols(df, ["lo", "hi"], into="ci", sep="-")
spd.replace(df, {"old": "new"}, columns=["status"])
spd.slice(df, where={"status": "ok"})
spd.sort(df, by="score", desc=True)
```

## Statistical column finders

```python
spd.find_pval(df)        # column matching p, p-value, pval, pvalue, etc.
spd.find_indi(df)        # indicator/dummy columns
```

## Convenience numerics

```python
spd.to_numeric(df, errors="coerce")
spd.round(df, decimals=3)
spd.get_unique(df, columns=["subject"])
```

## Suppress the chained-assignment chatter

```python
with spd.ignore_setting_with_copy_warning():
    df["new"] = df["old"] * 2
```

## When to use

- ✅ Coercing user input of unknown type into a DataFrame at API boundary
- ✅ Reshaping experimental data between long/wide/xyz forms
- ✅ Columns whose semantic role (p-value, indicator) is convention-based
- ❌ Heavy SQL-like operations — use pandas/duckdb directly

## See also

- `scitex-stats` — statistical tests that consume `find_pval` outputs
- `scitex-plt` — plotting hooks that pair with `to_xy`/`from_xyz`

<!-- EOF -->
