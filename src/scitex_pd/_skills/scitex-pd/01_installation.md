---
description: |
  [TOPIC] Installation
  [DETAILS] pip install scitex-pd. Pure-Python, depends on numpy + pandas + scitex-types.
tags: [scitex-pd-installation]
---

# Installation

## Standard

```bash
pip install scitex-pd
```

Pulls `numpy`, `pandas`, and `scitex-types`. No system deps.

## Verify

```bash
python -c "import scitex_pd as spd; print(spd.__version__)"
python -c "from scitex_pd import force_df, find_pval, mv; print('ok')"
```

## Editable install (development)

```bash
git clone https://github.com/ywatanabe1989/scitex-pd
cd scitex-pd
pip install -e '.[dev]'
```

## Umbrella alternative

`pip install scitex` exposes the same module as `scitex.pd`. Use either
`import scitex_pd as spd` (standalone) or `import scitex.pd as spd`
(umbrella) — they are equivalent at runtime.
