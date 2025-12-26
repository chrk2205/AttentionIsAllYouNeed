.PHONY: lint
lint:
	uv run ruff check --fix

.PHONY: train
train:
	uv run python -m src.train --input_file test