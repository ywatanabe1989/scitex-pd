---
description: |
  [TOPIC] Quick start
  [DETAILS] Smallest example — coerce arbitrary input to DataFrame, reorder columns, find p-value column.
tags: [scitex-pd-quick-start]
---

# Quick Start

## Coerce + reshape + reorder

```python
import scitex_pd as spd

# Coerce anything to a DataFrame
df = spd.force_df({"score": [1, 2, 3], "label": ["a", "b", "c"]})

# Reorder columns by name
spd.mv_to_first(df, "label")

# Find the p-value column by convention (matches p, pval, p-value, ...)
pcol = spd.find_pval(results_df)
```

## Long ↔ wide ↔ xyz

```python
long = spd.melt_cols(df, id_vars=["subject"], value_pattern=r"trial_\d+")
wide = spd.from_xyz(long_xyz_df)
xy   = spd.to_xy(coord_df)
```

## Suppress chained-assignment chatter

```python
with spd.ignore_setting_with_copy_warning():
    df["new"] = df["old"] * 2
```

## Next

- [03_python-api.md](03_python-api.md) — full public surface
- [SKILL.md](SKILL.md) — overview + when-to-use
