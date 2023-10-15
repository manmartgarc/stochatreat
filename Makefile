.DEFAULT_GOAL := all

# taken from here: https://dwmkerr.com/makefile-help-command/
.PHONY: help
help: # Show help for each of the Makefile recipes
	@grep -E '^[a-zA-Z0-9 -]+:.*#'  Makefile | sort | while read -r l; do printf "\033[1;32m$$(echo $$l | cut -f 1 -d':')\033[00m:$$(echo $$l | cut -f 2- -d'#')\n"; done


all: clean lint format test dist # Runs all recipes.

clean: clean-build clean-pyc clean-test # Runs all clean recipes.

clean-build: # Cleans all build files.
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: # Cleans all python cache files.
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: # Cleans all test files.
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

dist: clean # Builds source and wheel package.
	python -m build
	ls -lth dist/

lint: lint/ruff # Runs all linting.

lint/ruff: # Runs ruff linting.
	ruff src tests

release: dist # Releases a new version to PyPi.
	twine upload dist/*

format: format/black # Runs all format checks.

format/black: # Runs black format checks.
	black --check src tests

test: test/mypy test/pytest # Runs all tests.

test/pytest: # Runs pytest tests.
	pytest

test/mypy: # Runs mypy tests.
	mypy