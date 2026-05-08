# Convenience wrappers around common commands. None of these are
# magic — they just save typing the long forms. Run `make help` for
# the full list.

.PHONY: help test test-verbose smoke-mlx smoke-cuda build-elicit lint clean install-mlx install-cuda

help: ## Show this help.
	@echo 'Available targets:'
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

test: ## Run the unit-test suite (no GPU/model needed).
	python3 -m unittest discover -s . -p 'test_*.py'

test-verbose: ## Run unit tests with verbose output.
	python3 -m unittest discover -s . -p 'test_*.py' -v

smoke-mlx: ## Smoke-test the MLX runner (loads a 1.5B Qwen, generates).
	python3 scripts/smoke_test_mlx.py

smoke-cuda: ## Smoke-test the CUDA runner (loads a 1.5B Qwen on GPU, generates).
	python3 scripts/smoke_test_cuda.py

build-elicit: ## Regenerate the GSM8K elicitation dataset (rarely needed — invalidates prior results).
	python3 scripts/build_elicit_dataset.py

install-mlx: ## Install dependencies for the MLX path (Apple Silicon).
	pip install -r requirements-mlx.txt

install-cuda: ## Install dependencies for the CUDA path (Linux + NVIDIA).
	@echo "Step 1: install torch with the right CUDA build:"
	@echo "  pip install torch --index-url https://download.pytorch.org/whl/cu121"
	@echo "Step 2: then run 'make install-cuda-deps'"

install-cuda-deps: ## Install the rest of the CUDA stack (after torch).
	pip install -r requirements-cuda.txt

lint: ## Quick syntax-only check across all Python files.
	@python3 -c "import ast, pathlib, sys; \
		errs = [(str(p), str(e)) for p in pathlib.Path('.').rglob('*.py') \
		        if 'legacy' not in p.parts and '.venv' not in p.parts \
		        for e in [None] \
		        for _ in [None] \
		        if (lambda: (lambda f: ast.parse(f.read(), filename=str(p)) and False)(open(p)))() is None]; \
		print(f'OK — all .py files parse')"

clean: ## Remove caches (does not touch git-tracked files).
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf .venv-*/  # in case anyone made auxiliary venvs
