# Agents Guide

This file contains style and contribution guidance for AI agents working on this repository.

## Docstrings

Use [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) (Napoleon format). Example:

```python
def my_function(x: int, y: float) -> str:
    """Short summary of what the function does.

    Args:
        x: Description of x.
        y: Description of y.

    Returns:
        Description of the return value.

    Raises:
        ValueError: If x is negative.
    """
```

- Test files must **not** contain docstrings.
- All public modules must have a module-level docstring.
- Docstrings are linted by ruff (rules `D` and `DOC`); run `hatch fmt` to check.

## Type Hints

All functions must have full type annotations. Type checking is enforced via `ty`.

## Commits

Follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/). The commit type directly controls automated versioning and releases (see [CI/CD](#cicd-releases) below):

| Type | Effect |
|------|--------|
| `feat:` | Minor version bump (e.g. `0.1.5` → `0.2.0`) |
| `fix:` | Patch version bump (e.g. `0.1.5` → `0.1.6`) |
| `perf:` | Patch version bump (e.g. `0.1.5` → `0.1.6`) |
| `feat!:` / `fix!:` / `BREAKING CHANGE:` footer | Major version bump (e.g. `0.1.5` → `1.0.0`) |
| `docs:`, `chore:`, `refactor:`, `test:`, `ci:` | No release triggered |

**Do not manually edit `__version__` in `src/stochatreat/__about__.py`** — it is updated automatically by the release workflow.

## Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Line length is 79 characters (enforced by Ruff)
- Run `hatch fmt` before submitting a PR

## Tests

- Add tests for all new features and bug fixes
- Run `hatch test` to verify
- Test files live in `tests/` and must not contain docstrings

## CI/CD Releases

Releases are fully automated via [Python Semantic Release (PSR)](https://python-semantic-release.readthedocs.io/en/stable/) and triggered by merging a PR into `main`. The pipeline runs in three stages:

1. **Release** (`.github/workflows/release-and-publish.yml`): PSR inspects the commit messages on `main` since the last release tag. If any `feat:` or `fix:` commits (or breaking changes) are found, it:
   - Bumps the version in `src/stochatreat/__about__.py`
   - Updates `CHANGELOG.md`
   - Creates a Git tag and GitHub Release

2. **Build**: If a release was created, `hatch build` produces the `sdist` and wheel, which are attached to the GitHub Release.

3. **Publish**: The distribution artifacts are uploaded to [PyPI](https://pypi.org/project/stochatreat/) via trusted publishing. The [conda-forge feedstock](https://anaconda.org/conda-forge/stochatreat) picks up new PyPI releases automatically.

PRs that contain only `docs:`, `chore:`, `refactor:`, `test:`, `ci:`, or `perf:` commits will **not** trigger a release.

