format:
	uv run ruff format .

lint:
	uv run ruff check . --fix --preview

type-check:
	uv run mypy .

check: format lint type-check

sync:
	uv run wandb sync --sync-all
