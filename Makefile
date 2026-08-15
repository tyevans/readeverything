UV := uv run --all-extras

.PHONY: check lint types arch sec test fmt hooks

check: lint types arch sec test

# Install the git hook. Runs everything in `check` except the suite and
# pip-audit -- see .pre-commit-config.yaml for why those two stay in CI.
hooks:
	$(UV) pre-commit install

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
	$(UV) pytest --cov --cov-report=term-missing
