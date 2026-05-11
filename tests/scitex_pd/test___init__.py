#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for `scitex_pd/__init__.py` — version resolution + re-export surface."""

from __future__ import annotations

import importlib
from importlib.metadata import PackageNotFoundError
from unittest import mock


class TestVersionResolution:
    def test_version_is_string(self):
        import scitex_pd

        assert isinstance(scitex_pd.__version__, str)
        assert scitex_pd.__version__  # non-empty

    def test_package_not_found_falls_back_to_local(self):
        """When `importlib.metadata.version()` cannot locate the
        installed distribution (e.g. running from a checkout that was
        never installed), `__init__.py` must fall back to a local
        sentinel instead of raising at import time."""
        import scitex_pd

        with mock.patch(
            "importlib.metadata.version",
            side_effect=PackageNotFoundError("scitex-pd"),
        ):
            reloaded = importlib.reload(scitex_pd)
            assert reloaded.__version__ == "0.0.0+local"

        # Restore the real-version state for downstream tests in this
        # process — reload again so `_v()` runs against the unmocked
        # metadata module.
        importlib.reload(scitex_pd)


class TestPublicSurface:
    def test_all_symbols_are_importable(self):
        import scitex_pd

        for name in scitex_pd.__all__:
            assert hasattr(scitex_pd, name), f"missing public symbol: {name}"
