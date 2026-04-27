# scitex-pd

<!-- scitex-badges:start -->
[![PyPI](https://img.shields.io/pypi/v/scitex-pd.svg)](https://pypi.org/project/scitex-pd/)
[![Python](https://img.shields.io/pypi/pyversions/scitex-pd.svg)](https://pypi.org/project/scitex-pd/)
[![Tests](https://github.com/ywatanabe1989/scitex-pd/actions/workflows/test.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-pd/actions/workflows/test.yml)
[![Install Test](https://github.com/ywatanabe1989/scitex-pd/actions/workflows/install-test.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-pd/actions/workflows/install-test.yml)
[![Coverage](https://codecov.io/gh/ywatanabe1989/scitex-pd/graph/badge.svg)](https://codecov.io/gh/ywatanabe1989/scitex-pd)
[![Docs](https://readthedocs.org/projects/scitex-pd/badge/?version=latest)](https://scitex-pd.readthedocs.io/en/latest/)
[![License: AGPL v3](https://img.shields.io/badge/license-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
<!-- scitex-badges:end -->


Pandas helpers extracted from the [SciTeX](https://github.com/ywatanabe1989/scitex-python) ecosystem as a standalone package.

## Install

```bash
pip install scitex-pd
```

## API

```python
import scitex_pd as pd_

pd_.force_df(data)              # Coerce dict / Series / list / scalar → DataFrame
pd_.from_xyz(df, x, y, z)       # Long → wide pivot
pd_.to_xy(df) / pd_.to_xyz(df)  # Wide → long
pd_.find_pval(df)               # Locate p-value columns
pd_.find_indi(df, mask)         # Index lookup
pd_.get_unique(df)              # Unique value summary
pd_.merge_columns(df, [...], "out")
pd_.melt_cols(df, [...])
pd_.mv(df, col, position=-1)    # Move a column
pd_.replace(df, mapping)
pd_.round(df, ndigits=2)
pd_.slice(df, ...)
pd_.sort(df, ...)
pd_.to_numeric(df)
pd_.ignore_setting_with_copy_warning()
```

## Status

Standalone fork of `scitex.pd`. Deps: numpy, pandas, scitex-types (for `is_listed_X`).
The umbrella package's `scitex.pd` import path is preserved via a
`sys.modules`-alias bridge. 347/347 tests pass.

## License

AGPL-3.0-only (see [LICENSE](./LICENSE)).
