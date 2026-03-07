.PHONY: build clean test

build:
	bash scripts/build_core.sh

clean:
	rm -rf build/

test:
	python -m pytest tests/ -v

install:
	pip install -e ".[dev]"
