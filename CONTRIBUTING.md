<!-- Inspired by https://github.com/burnash/gspread/blob/master/.github/CONTRIBUTING.md -->
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
- Subscribe to the [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/) specification for commit messages. Based on whether you add a new feature, fix a bug, or make a breaking change, please prefix your commit messages with `feat:`, `fix:`, or `BREAKING CHANGE:` respectively. This will not only help in maintaining a clear and consistent project history but also allow automated tools to generate changelogs and manage versioning effectively, including releases.

- Please follow [Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/).

## Setting up development environment

You can install the development environment, i.e. all the dependencies required to run all tests and checks that are run when you submit a PR, by following these steps:

1. [Install](https://hatch.pypa.io/1.9/install/#installation) `hatch`.
2. Clone the repository:

    ```bash
    git clone https://github.com/manmartgarc/stochatreat
    cd stochatreat
    ```

3. Confirm `hatch` picked up the project:

    ```bash
    hatch status
    ```

## Tests

To run tests in the default environment:

```bash
hatch test
```

To run tests in all environments:

```bash
hatch run all:test
```

## Format

When submitting a PR, the CI will run `make format` and also `make lint` to check the format of the code. You can run this locally by running:

```bash
hatch fmt
```

## Release

Based on your conventional commit tags, the CI process will determine whether or not a new release is required. If it does, it will handle this for you by:

- Creating a new tag
- Creating a new release
- Building the release artifacts
- Attaching these artifacts to the release
- Doing the same for PyPi
