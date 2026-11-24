.PHONY: test lint web-build

test:
	python scripts/check.py

lint:
	python scripts/check.py --lint-only

web-build:
	cd web && npm ci && npm run build && npm run lint
