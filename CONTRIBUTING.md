# Contributing

Thanks for helping improve `symbolic-neural-networks`.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Coding conventions

- Keep symbols and argument names explicit and consistent.
- Prefer readable, small helper functions for reusable symbolic operations.
- Keep architecture builders in `symbolic_neural_networks/symbolic.py` and return a `SymbolicModel`.
- Export new public symbols through `__all__` in `symbolic_neural_networks/symbolic.py` and use package imports in `symbolic_neural_networks/__init__.py`.

## Adding a new model family

1. Implement `build_<arch>_model(...)` in `symbolic_neural_networks/symbolic.py`.
2. Add argument parsing and dispatch logic in `symbolic_neural_networks/cli.py`.
3. Add at least one small example case in `examples/symbolic_inspection.py`.
4. Update `README.md` API section and architecture list.

## Submitting changes

Use focused commits and clear messages (for example: `feat: add conv model builder`, `fix: simplify gcn degree handling`).

## Compatibility

If you change public package behavior, update:

- `README.md`
- `CHANGELOG.md`
- any relevant examples and exports

## Release process

Release flow is automated via `.github/workflows/release.yml`:

1. Update package version in `pyproject.toml`.
2. Add a matching section in `CHANGELOG.md` (e.g. `## [x.y.z]`).
3. Commit changes.
4. Push a tag `vX.Y.Z` (for example `v0.1.1`) to trigger release build + publish.

For dry-run verification without publishing, run the workflow manually with `publish=false`.
For publishing on manual runs, run with `publish=true` and set `PYPI_API_TOKEN` secret in repository settings.
