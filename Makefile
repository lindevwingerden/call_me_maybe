.PHONY: all install run debug clean lint lint-strict

all: install run

install:
	uv sync

run: install
	uv run python -m src

debug: install
	uv run python -m pdb -m src

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +

lint: install
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
	uv run flake8 .

lint-strict: install
	uv run mypy . --strict
	uv run flake8 .
