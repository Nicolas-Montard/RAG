run:
	uv run -m index

install:
	uv sync

debug:
	uv run python -m pdb -m src index

clean:
	rm -rf data/output/*
	rm -rf data/processed/*
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +

lint:
	uv run find src -name "*.py" -not -path "*/.*" -exec flake8 {} +
	uv run mypy src --warn-return-any --warn-unused-ignores \
	--ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
