.DEFAULT_GOAL := help
SHELL         := /bin/bash

VENV   := .venv
PYTHON ?= python3
PY     := $(VENV)/bin/python
PIP    := $(VENV)/bin/pip
EXTRAS ?= dev,train,embed

UNAME_S    := $(shell uname -s)
SHARED_EXT := $(if $(filter Darwin,$(UNAME_S)),dylib,so)
CORE_LIB   := build/libocgcore.$(SHARED_EXT)
CORE_SRC   := $(shell find third_party/ygopro-core \
                \( -name '*.cpp' -o -name '*.c' -o -name '*.h' -o -name '*.hpp' \) 2>/dev/null)

.PHONY: help setup submodules install install-minimal install-web \
        build build-force build-web build-mud assets \
        test test-py test-web lint format hooks clean clean-all

help:            ## List targets
	@grep -hE '^[a-z][a-z-]*:.*?## ' $(MAKEFILE_LIST) \
	  | awk -F':.*## ' '{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# ── Lifecycle ───────────────────────────────────────────────────────────────
# Steps are separate recipe lines, not prerequisites, so they run in order even
# under -j (a target's prerequisites have no guaranteed order).
setup:           ## Set up a fresh clone (MINIMAL=1 skips torch, NO_WEB=1 skips JS)
	$(MAKE) submodules
	$(MAKE) $(if $(MINIMAL),install-minimal,install)
	$(MAKE) build
	$(MAKE) assets
	$(if $(NO_WEB),,$(MAKE) install-web)
	@echo "Done. Run 'make help' to see what else is available."

submodules:      ## Init and update git submodules
	git submodule update --init --recursive

# ── Install ─────────────────────────────────────────────────────────────────
install:         ## Install Python deps into .venv (override EXTRAS=)
	test -d $(VENV) || $(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e ".[$(EXTRAS)]"

install-minimal: ## Install Python deps without torch (~250 MB vs ~1.3 GB)
	$(MAKE) install EXTRAS=dev

install-web:     ## Install JS deps (skips if node/pnpm absent)
	bash scripts/fetch_web_deps.sh

# ── Build ───────────────────────────────────────────────────────────────────
build: $(CORE_LIB)  ## Compile libocgcore (no-op when up to date)

$(CORE_LIB): $(CORE_SRC) scripts/build_core.sh
	bash scripts/build_core.sh

build-force:     ## Recompile libocgcore from scratch
	rm -f $(CORE_LIB)
	$(MAKE) build

build-web:       ## Compile the web bundle (run install-web first)
	bash scripts/build_web.sh

build-mud:       ## Build the third-party MUD server
	bash scripts/build_mud_server.sh

assets:          ## Download cards.cdb + strings.conf (FORCE=1 to refresh)
	bash scripts/fetch_assets.sh $(if $(FORCE),--force,)

# ── Verify ──────────────────────────────────────────────────────────────────
test: test-py test-web  ## Run both test suites (Python + web)

test-py:         ## Run pytest
	$(PY) -m pytest tests/ -v

test-web:        ## Run vitest (skips without node; STRICT_WEB=1 to fail)
	bash scripts/test_web.sh

lint:            ## Lint with ruff + eslint
	$(VENV)/bin/ruff check .
	bash scripts/lint_web.sh

format:          ## Format with ruff + prettier
	$(VENV)/bin/ruff format .
	bash scripts/format_web.sh

hooks:           ## Install pre-commit hooks
	@git config --get core.hooksPath >/dev/null 2>&1 \
	  && echo "hooks already active via core.hooksPath" \
	  || $(VENV)/bin/pre-commit install

# ── Clean ───────────────────────────────────────────────────────────────────
clean:           ## Remove build/
	rm -rf build/

clean-all: clean ## Also remove venv, JS deps, web bundle, MUD server
	rm -rf .venv yugioh_web/node_modules yugioh_web/dist third_party/yugioh-game
