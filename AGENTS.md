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
    """
```

## Type Hints

All functions must have full type annotations. Type checking is enforced via `ty`.

## Commits

Follow [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/):
- `feat:` for new features
- `fix:` for bug fixes
- `BREAKING CHANGE:` for breaking changes

## Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Line length is 79 characters (enforced by Ruff)
- Run `hatch fmt` before submitting a PR

## Tests

- Add tests for all new features and bug fixes
- Run `hatch test` to verify
