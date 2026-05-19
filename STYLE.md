# Code style

Project conventions beyond what ruff enforces (ruff config lives in `[tool.ruff]` in `pyproject.toml`).

## Ruff

- Line length 80 (72 for docstrings and comments).
- Single quotes.
- Rules: `E, F, W, I, G`.

## Typing

- Annotate function arguments and returns. Modern union syntax (`X | None`), not `typing.Optional`.
- Explicit `-> None`.
- Don't annotate obvious locals. Do annotate empty collections, `Any`-returning calls, `None`-initialized vars.

## Errors

- Validate input at the boundary. `raise TypeError` / `ValueError` with the offending type or value in the message.

## Logging

- `logging.basicConfig` only inside `main()`. Module scope keeps just `logger = logging.getLogger(__name__)`.
- `%`-style placeholders, no f-strings (enforced by `G004`).

## Docstrings

- NumPy-style. Public functions only — private (`_name`) helpers don't get docstrings.
- Sections: one-line summary; `Parameters`; `Returns` (when non-None); `Raises` (when applicable).
- Blank line after every docstring.
- 72-char limit (enforced by `W505`).

## Imports

- `__all__` in every `__init__.py` that re-exports symbols.

## Formatting

- Only split parenthesized expressions when a single line would exceed 80 chars.
- Blank line before the final `return` in non-trivial functions.

## Tests

- Mirror source layout (`pyment/x/y.py` → `tests/x/test_y.py`).
- Testing private (`_name`) functions is fine.
- Assert messages follow `'Expected <subject> to <verb> <outcome>'`. The subject is the function or the value it returns. Use `return` for scalars/objects, `yield` for images/arrays. Reference parameter names rather than their values when the expected result is input-derived; state values explicitly only when they are hardcoded in the implementation. Append a condition clause (`for ...`, `when ...`) when the scenario needs disambiguation.
