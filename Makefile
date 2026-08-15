UV := uv run --all-extras

.PHONY: check lint types arch sec test fmt

check: lint types arch sec test

lint:
	$(UV) ruff check .
	$(UV) ruff format --check .

fmt:
	$(UV) ruff format .
	$(UV) ruff check --fix .

types:
	$(UV) mypy

arch:
	$(UV) lint-imports

sec:
	$(UV) bandit -q -c pyproject.toml -r src
	$(UV) pip-audit

test:
	$(UV) pytest
