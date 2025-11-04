format:
	uv run ruff format .

lint:
	uv run ruff check . --fix --preview

type-check:
	uv run mypy src 

check: format lint type-check

test:
	pytest -v tests

sync:
	uv sync --all-extras --all-groups

stubgen:
	stubgen -p blazefl.core -p blazefl.reproducibility --no-analysis -o src
