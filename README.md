# scitex-pd

<p align="center">
  <a href="https://scitex.ai">
    <img src="docs/scitex-logo-blue-cropped.png" alt="SciTeX" width="400">
  </a>
</p>

<p align="center"><b>Pandas helpers — coerce, reshape, find p-values, merge/melt columns, sort/slice/round.</b></p>

<p align="center">
  <a href="https://scitex-pd.readthedocs.io/">Full Documentation</a> · <code>pip install scitex-pd</code>
</p>

<!-- scitex-badges:start -->
<p align="center">
  <a href="https://pypi.org/project/scitex-pd/"><img src="https://img.shields.io/pypi/v/scitex-pd.svg" alt="PyPI"></a>
  <a href="https://pypi.org/project/scitex-pd/"><img src="https://img.shields.io/pypi/pyversions/scitex-pd.svg" alt="Python"></a>
  <a href="https://github.com/ywatanabe1989/scitex-pd/actions/workflows/test.yml"><img src="https://github.com/ywatanabe1989/scitex-pd/actions/workflows/test.yml/badge.svg" alt="Tests"></a>
  <a href="https://github.com/ywatanabe1989/scitex-pd/actions/workflows/install-test.yml"><img src="https://github.com/ywatanabe1989/scitex-pd/actions/workflows/install-test.yml/badge.svg" alt="Install Test"></a>
  <a href="https://codecov.io/gh/ywatanabe1989/scitex-pd"><img src="https://codecov.io/gh/ywatanabe1989/scitex-pd/graph/badge.svg" alt="Coverage"></a>
  <a href="https://scitex-pd.readthedocs.io/en/latest/"><img src="https://readthedocs.org/projects/scitex-pd/badge/?version=latest" alt="Docs"></a>
  <a href="https://www.gnu.org/licenses/agpl-3.0"><img src="https://img.shields.io/badge/license-AGPL_v3-blue.svg" alt="License: AGPL v3"></a>
</p>
<!-- scitex-badges:end -->

---

## Installation

```bash
pip install scitex-pd
```

## Quick Start

```python
import scitex_pd as pd_

pd_.force_df(data)              # Coerce dict / Series / list / scalar → DataFrame
pd_.from_xyz(df, x, y, z)       # Long → wide pivot
pd_.to_xy(df)                   # Wide → long
pd_.find_pval(df)               # Locate p-value columns
```

## 1 Interfaces

<details open>
<summary><strong>Python API</strong></summary>

<br>

```python
import scitex_pd as pd_

# Coerce / convert
pd_.force_df(data)
pd_.from_xyz(df, x, y, z)
pd_.to_xy(df)
pd_.to_xyz(df)
pd_.to_numeric(df)

# Find / inspect
pd_.find_pval(df)
pd_.find_indi(df, mask)
pd_.get_unique(df, "col")

# Reshape / restructure
pd_.merge_columns(df, [...], "out")
pd_.melt_cols(df, [...])
pd_.mv(df, col, position=-1)

# Transform
pd_.replace(df, mapping)
pd_.round(df, ndigits=2)
pd_.slice(df, ...)
pd_.sort(df, ...)

# Warnings
pd_.ignore_setting_with_copy_warning()
```

</details>

## Status

Standalone fork of `scitex.pd`. Deps: numpy, pandas, scitex-types (for `is_listed_X`).
The umbrella package's `scitex.pd` import path is preserved via a
`sys.modules`-alias bridge.

## Part of SciTeX

`scitex-pd` is part of [**SciTeX**](https://scitex.ai). Install via
the umbrella with `pip install scitex[pd]` to use as
`scitex.pd` (Python) or `scitex pd ...` (CLI).

>Four Freedoms for Research
>
>0. The freedom to **run** your research anywhere — your machine, your terms.
>1. The freedom to **study** how every step works — from raw data to final manuscript.
>2. The freedom to **redistribute** your workflows, not just your papers.
>3. The freedom to **modify** any module and share improvements with the community.
>
>AGPL-3.0 — because we believe research infrastructure deserves the same freedoms as the software it runs on.

## License

AGPL-3.0-only (see [LICENSE](./LICENSE)).

---

<p align="center">
  <a href="https://scitex.ai" target="_blank"><img src="docs/scitex-icon-navy-inverted.png" alt="SciTeX" width="40"/></a>
</p>
