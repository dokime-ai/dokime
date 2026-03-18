# Contributing to Dokime

Thank you for your interest in contributing to Dokime! This guide will help you get started.

## Development Setup

1. Fork and clone the repository:
```bash
git clone https://github.com/dokime-ai/dokime.git
cd dokime
```

2. Install in development mode:
```bash
pip install -e ".[dev]"
```

3. Install pre-commit hooks:
```bash
pre-commit install
```

4. Verify everything works:
```bash
pytest
ruff check src/ tests/
```

## Making Changes

1. Create a branch from `main`:
```bash
git checkout -b your-feature-name
```

2. Make your changes, ensuring:
   - All new code has type hints on public APIs
   - New filters inherit from `dokime.core.filters.Filter`
   - Optional dependencies use lazy imports with clear error messages
   - Tests are added for new functionality

3. Run the checks:
```bash
ruff check src/ tests/       # lint
ruff format src/ tests/       # format
pytest                        # test
mypy src/dokime/              # type check (optional)
```

4. Commit and push:
```bash
git add .
git commit -m "Add your feature description"
git push origin your-feature-name
```

5. Open a Pull Request against `main`.

## Code Style

- **Formatter/Linter:** Ruff (configured in `pyproject.toml`)
- **Line length:** 119 characters
- **Quotes:** Double quotes
- **Imports:** Sorted by ruff (isort-compatible)
- **Type hints:** Required on all public function signatures

## Adding a New Filter

1. Create your filter class in the appropriate module under `src/dokime/`
2. Inherit from `dokime.core.filters.Filter`
3. Implement `filter(self, sample: dict) -> bool` and `name(self) -> str`
4. Add tests in `tests/`
5. If it requires new dependencies, add them as an optional extra in `pyproject.toml`

## Reporting Issues

- Use the [Bug Report](.github/ISSUE_TEMPLATE/bug_report.yml) template for bugs
- Use the [Feature Request](.github/ISSUE_TEMPLATE/feature_request.yml) template for new ideas

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
