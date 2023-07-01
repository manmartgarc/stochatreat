.DEFAULT_GOAL := all

all: clean lint style test dist

clean: clean-build clean-pyc clean-test

clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test:
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

dist: clean
	python -m build
	ls -lth dist/

lint: lint/ruff

lint/ruff:
	ruff src tests

release: dist
	twine upload dist/*

style: style/black

style/black:
	black --check src tests

test: test/mypy test/pytest

test/pytest:
	pytest

test/mypy:
	mypy