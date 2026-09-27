# Releasing ZTPCraft

## One-time account setup

Create accounts on PyPI and TestPyPI and enable their required two-factor
authentication. Configure a pending Trusted Publisher for a new project (or add a
publisher to the existing project if you own it):

| Field | PyPI | TestPyPI |
| --- | --- | --- |
| Project name | `ztpcraft` | `ztpcraft` |
| GitHub owner | `ZhaoTianPu` | `ZhaoTianPu` |
| Repository | `ZTPCraft` | `ZTPCraft` |
| Workflow filename | `publish.yml` | `publish.yml` |
| Environment | `pypi` | `testpypi` |

Create matching GitHub repository environments. Configure required reviewers for
`pypi` if you want an approval before publication. No PyPI API token or repository
secret is needed. A pending publisher does not reserve the package name.

Official instructions: [new projects](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
and [publishing](https://docs.pypi.org/trusted-publishers/using-a-publisher/).

## Check a release locally

Use a clean checkout and an isolated environment:

```sh
python -m venv /tmp/ztpcraft-build
/tmp/ztpcraft-build/bin/python -m pip install build twine
/tmp/ztpcraft-build/bin/python -m build
/tmp/ztpcraft-build/bin/python -m twine check --strict dist/*
python -m venv /tmp/ztpcraft-wheel-test
/tmp/ztpcraft-wheel-test/bin/python -m pip install dist/*.whl pytest
/tmp/ztpcraft-wheel-test/bin/python -m pip check
/tmp/ztpcraft-wheel-test/bin/python -I tools/check_install.py
/tmp/ztpcraft-wheel-test/bin/python -I -m pytest --import-mode=importlib tests/test_fluxoid_matrix_rates.py tests/test_fluxoid_transition_rates.py tests/test_fgr.py tests/test_quantum_noise.py
```

Use fresh temporary directory names if these already exist. The default build
creates a source archive, then builds the wheel from that archive. The installation
check verifies that Python loads the installed package, its Cython extension, and
the rate API without importing JAX or ninatool. Tests run against the installed wheel.

## Publish

1. Set a new, unused version in `pyproject.toml`; the current version is `0.2`.
   Commit and push the release changes. Ordinary pushes and pull requests only build.
2. In GitHub Actions, run **Build and publish distributions**, selecting
   `testpypi`, to rehearse publication. `none` only builds and tests.
3. Inspect the results for all three platforms. Download the TestPyPI wheel without
   dependencies into a fresh folder, then install that local wheel normally so its
   dependencies come from PyPI:
   ```sh
   python -m pip download --no-deps --only-binary=:all: --index-url https://test.pypi.org/simple/ ztpcraft==0.2 -d test-wheel
   python -m pip install test-wheel/*.whl
   ```
   Replace `0.2` with the version being released and use a fresh virtual environment.
4. Create a GitHub release with tag `v0.2` (or `v` plus the exact new version).
   Publishing that release builds and tests the distributions and uploads to PyPI.
   Creating a tag alone does not publish.
5. Verify `python -m pip install ztpcraft==0.2` in a fresh environment and run the
   installation check. Share that exact version with collaborators for reproducibility.

PyPI does not allow uploaded filenames to be reused. Fix a published package by
bumping the version and creating a new release.

The workflow covers CPython 3.10–3.12, Linux x86_64, and macOS arm64/x86_64.
It builds each wheel from the source archive and tests it in isolation. Windows,
Linux ARM, and newer Python versions are not in the current wheel test matrix.
