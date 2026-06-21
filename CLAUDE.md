# CLAUDE.md

## Code Style

All code must pass the pre-commit hooks defined in `.pre-commit-config.yaml` before committing:

```bash
pre-commit run --all-files
```

Formatting and linting are governed by `pyproject.toml`:

| Tool | Role | Key settings |
|------|------|--------------|
| `ruff-format` | Formatter (Black drop-in) | line-length 120, double quotes |
| `ruff lint` | Linter | E, W, F, I, B, C4, UP rule sets |
| `pyright` | Type checker | `standard` mode, Python 3.12 |

Run the full suite manually:

```bash
uv run ruff format src/
uv run ruff check src/
uv run pyright src/
uv run pytest
```

## Type Hints

Every function signature and every variable declaration must carry an explicit type annotation.
Avoid `Any`; if unavoidable, add a short inline comment explaining why.

```python
def filter_raw(
    raw: mne.io.Raw,
    l_freq: float = 0.1,
    notch_freqs: list[float] | None = None,
) -> mne.io.Raw:
    threshold: float = 100e-6
    ...
```

## Docstrings

All functions require a minimal [Google-style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) docstring:
a one-line summary, `Args:`, and `Returns:`. Omit `Returns:` for `None`-returning functions.
Omit `Raises:` unless the function explicitly raises for a documented condition.

```python
def load_raw(path: Path, subject_id: str) -> mne.io.Raw:
    """Load BrainVision file and patch internal filename mismatch.

    Args:
        path: Path to the .vhdr file.
        subject_id: Subject identifier used for logging.

    Returns:
        Continuous raw EEG data.
    """
```

## README.md Updates

Every change to `README.md` must:

1. Update the `Last updated:` date in the file header to today's date in `YYYY-MM-DD` format.
2. Add a new entry to the `## Changelog` section (newest first):

```markdown
## Changelog

### YYYY-MM-DD
- Brief description of what changed.
```
