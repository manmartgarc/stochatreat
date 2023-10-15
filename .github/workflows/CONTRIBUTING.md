# Contributing Guide

- Check the [GitHub Issues](https://github.com/manmartgarc/stochatreat/issues) for open issues that need attention.
- Follow the [How to submit a contribution](https://opensource.guide/how-to-contribute/#how-to-submit-a-contribution) Guide.

- Make sure unit tests pass. Please read how to run unit tests [below](#tests).

- If you are fixing a bug:
  - If you are resolving an existing issue, reference the issue ID in a commit message `(e.g., fixed #XXXX)`.
  - If the issue has not been reported, please add a detailed description of the bug in the Pull Request (PR).
  - Please add a regression test case to check the bug is fixed.

- If you are adding a new feature:
  - Please open a suggestion issue first.
  - Provide a convincing reason to add this feature and have it greenlighted before working on it.
  - Add tests to cover the functionality.

- Please follow [Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/).

## Setting up development environment

You can install the development environment, i.e. all the dependencies required to run all tests and checks that are run when you submit a PR, by running. There are two main ways of doing this, one is to do it without cloning the repository, and the other is to clone the repository and then install the dependencies.

### Without cloning the repository

```bash
pip install "stochatreat[dev] @ git+https://github.com/manmartgarc/stochatreat"
```

### Cloning the repository

```bash
git clone https://github.com/manmartgarc/stochatreat
cd stochatreat
pip install -e .[dev]
```

## Tests

To run tests run:

```bash
make test
```

## Format

When submitting a PR, the CI will run `make format` and also `make lint` to check the format of the code. You can run this locally by running:

```bash
make format lint
```

## Release

New release system:

- Update version number in [`gspread/__init__.py`](../gspread/__init__.py).
- Get changelog from drafting a new [GitHub release](https://github.com/burnash/gspread/releases/new) (do not publish, instead cancel.)
- Add changelog to [`HISTORY.rst`](../HISTORY.rst).
- Commit the changes as `Release vX.Y.Z` (do not push yet.)
- Run `tox -e lint,py,build,doc` to check build/etc.
- Push the commit. Wait for the CI to pass.
- Add a tag `vX.Y.Z` to the commit locally. This will trigger a new release on PyPi, and make a release on GitHub.
- View the release on [GitHub](https://github.com/burnash/gspread/releases) and [PyPi](https://pypi.org/project/gspread/)!