.PHONY: test lint web-build

test:
	python -m ruff check .
	python -m pytest -q

lint:
	python -m ruff check .

web-build:
	cd web && npm ci && npm run build && npm run lint
