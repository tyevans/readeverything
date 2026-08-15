# Releasing

The distribution, the import package, and the repository are all
`readeverything`.

## How it works

Pushing a `v*` tag runs `.github/workflows/release.yml`, which:

1. checks the tag matches `version` in `pyproject.toml` (and fails first if not),
2. runs the whole CI workflow,
3. builds an sdist and wheel with `uv build` and smoke-tests the wheel by
   installing it into a clean venv and importing through it,
4. publishes to PyPI via Trusted Publishing (OIDC, `pypi` environment),
5. creates a GitHub Release with the CHANGELOG section for that version.

| Tag        | Destination | GitHub Release |
|------------|-------------|----------------|
| `v0.1.0`   | PyPI        | normal         |
| `v0.1.0a1` | PyPI        | pre-release    |
| `v0.1.0b1` | PyPI        | pre-release    |
| `v0.1.0rc1`| PyPI        | pre-release    |

TestPyPI is not wired in. Pre-release versions are invisible to a plain
`pip install` anyway, so an alpha on PyPI reaches only people who ask for it
by version or pass `--pre`.

## One-time setup

On PyPI, add a **pending publisher** for the project before the first release
(https://pypi.org/manage/account/publishing/):

- PyPI project name: `readeverything`
- Owner: `tyevans`
- Repository: `readeverything`
- Workflow: `release.yml`
- Environment: `pypi`

Then create a GitHub environment named `pypi` in the repository settings.
Adding a required reviewer to it is worth doing: it turns every publish into
something a human clicks.

## Cutting a release

```bash
# 1. Version and changelog
#    - pyproject.toml: version = "X.Y.Z"
#    - CHANGELOG.md: move [Unreleased] entries under ## [X.Y.Z] - YYYY-MM-DD
#      and update the link refs at the bottom

# 2. Verify locally — the same gate CI runs
#    (`make hooks` installs the pre-commit subset of it once, per clone)
make check

# 3. Commit
git add pyproject.toml CHANGELOG.md
git commit -m "chore: prepare release X.Y.Z"
git push origin main

# 4. Tag
git tag vX.Y.Z
git push origin vX.Y.Z
```

Then watch the Release workflow, and confirm the version on
https://pypi.org/project/readeverything/.

Install a pre-release (they are invisible to a plain `pip install`):

```bash
pip install readeverything==0.2.0a1
```

## When something goes wrong

**Tag/version mismatch, or CI red after tagging.** Nothing has been published
— the validate and CI jobs run before build. Fix on `main`, then move the tag:

```bash
git tag -d vX.Y.Z && git push origin :refs/tags/vX.Y.Z
git tag vX.Y.Z && git push origin vX.Y.Z
```

**Publish failed after upload.** A version on PyPI cannot be replaced or
reused, even after a delete. Increment (`X.Y.Za2`, or `X.Y.Z+1`) and cut
again; do not try to reuse the number.

**A bad release is already out.** Yank it on PyPI rather than deleting it —
yanking leaves existing pins working while keeping new resolutions off it —
and ship the fix as a new version.
