---
name: scitex-pd
description: |
  [WHAT] pandas helpers extracted from the SciTeX ecosystem.
  [WHEN] Use when working with scitex-pd APIs or when the user mentions scitex.pd..
  [HOW] `import scitex_pd` then call `force_df(x)`.
tags: [scitex-pd]
primary_interface: python
interfaces:
  python: 3
  cli: 0
  mcp: 0
  skills: 2
  hook: 0
  http: 0
canonical-location: scitex-pd/src/scitex_pd/_skills/scitex-pd/SKILL.md
---


> **Interfaces:** Python ⭐⭐⭐ (primary) · CLI — · MCP — · Skills ⭐⭐ · Hook — · HTTP —

# scitex-pd

Conventional pandas helpers — replaces the boilerplate that everyone
re-writes when massaging dataframes for analysis.

## Sub-skills

- [01_installation.md](01_installation.md) — pip install + verify
- [02_quick-start.md](02_quick-start.md) — coerce + reshape + finders
- [03_python-api.md](03_python-api.md) — full public surface

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
