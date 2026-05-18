#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for `scitex_pd/__init__.py` — version resolution + re-export surface."""

from __future__ import annotations

import importlib
import importlib.metadata as _metadata
from importlib.metadata import PackageNotFoundError


class TestVersionResolution:
    def test_version_attribute_is_string_type(self):
        # Arrange
        import scitex_pd
        # Act
        version = scitex_pd.__version__
        # Assert
        assert isinstance(version, str)

    def test_version_attribute_is_non_empty_string(self):
        # Arrange
        import scitex_pd
        # Act
        version = scitex_pd.__version__
        # Assert
        assert version

    def test_package_not_found_falls_back_to_local_sentinel(self):
        """When `importlib.metadata.version()` cannot locate the
        installed distribution (e.g. running from a checkout that was
        never installed), `__init__.py` must fall back to a local
        sentinel instead of raising at import time."""
        # Arrange
        import scitex_pd

        original_version = _metadata.version

        def _raising_version(name):
            raise PackageNotFoundError(name)

        _metadata.version = _raising_version
        try:
            # Act
            reloaded = importlib.reload(scitex_pd)
            local_version = reloaded.__version__
        finally:
            _metadata.version = original_version
            # Restore the real-version state for downstream tests in this
            # process — reload again so `_v()` runs against the unswapped
            # metadata module.
            importlib.reload(scitex_pd)
        # Assert
        assert local_version == "0.0.0+local"


class TestPublicSurface:
    def test_all_listed_symbols_are_importable_from_package(self):
        # Arrange
        import scitex_pd

        missing = [
            name for name in scitex_pd.__all__ if not hasattr(scitex_pd, name)
        ]
        # Act
        missing_count = len(missing)
        # Assert
        assert missing_count == 0, f"missing public symbols: {missing}"
