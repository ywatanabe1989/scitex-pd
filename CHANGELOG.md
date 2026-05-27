# Changelog

All notable changes to `scitex-pd` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.6]

- Fix tests: remove duplicate `from __future__ import annotations`.
- ci(docs): make _sphinx_html commit-back step non-fatal.
- ci(quality): replace ecosystem-clone template with single-package audit-all.
- tests: add subprocess-coverage wiring to conftest.
- fix(workflows): resync integrated release pipeline from scitex-dev v0.11.20.
- fix(workflows): standardize workflows to scitex-dev canonical set.

## [0.1.5]

- Build and release pipeline improvements.

## [0.1.4]

- fix(tests): clear PA-306 + PA-307 test-quality violations.
- docs(api): add Functionalities/IO/Dependencies triple to top-level docstring.
- docs(skills): drop PS-116 banned Interfaces callout from SKILL.md.
- chore(pyproject): drop banned author email.

## [0.1.3]

- ci: normalize workflow filenames + README badges (PS-164).
- test: raise coverage 86 % → 100 % with real branch tests.
- ci(newb): switch auth to NEWB_CLAUDE_CODE_CREDENTIALS_JSON.
- docs(readme): recommend `uv pip install <pkg>[all]`.

## [0.1.2]

- ci: sync GitHub Releases with PyPI publish.
- chore(deps): bump scitex-dev pin floor.
- archive(tests): add from_umbrella.tar.gz.
- ci(newb): add weekly doc-quality workflow.
- fix(_to_xy): import missing _mv helper.
- test: bulk rewrite `from scitex.pd` → `from scitex_pd` imports.
- docs(readme): add Demo + Architecture sections (PS141/PS142).
- fix(deps): drop umbrella `scitex` from optional-deps[dev] (PS139).
- docs: add CHANGELOG.md + CONTRIBUTING.md (audit-project PS134/PS135).
- test(audit): integrate audit-all into the test suite.
- docs(skills): add mandatory installation/quick-start/python-api leaves.
- docs(skills): adopt inline [WHAT]/[WHEN]/[HOW] marker standard.
- docs(README): PS133 badges below Full-Doc line.
- docs(readme): primary interface <details open> (PS131).
- ci: linguist-generated for _sphinx_html/.
- ci+docs: add canonical docs.yml + sphinx/requirements.txt (PS122/PS126).
- docs(readme): include CLI form in 'Part of SciTeX' (PS120).
- docs(readme): adopt canonical 'Part of SciTeX' one-liner.
- audit: clear all 8 PS warnings.
- fix(release-safety): opt-in publish-pypi.yml.
- fix(skills): strip trailing `<!-- EOF -->` (SK211).
- fix(api): drop private names from `__all__` (PA103).
- fix(api): PA501/PA201/PA203 hygiene.
- chore(version): switch `__version__` to importlib.metadata.

## [0.1.1]

- Initial CHANGELOG entry — see git log for prior history.
