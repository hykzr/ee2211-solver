# EE2211 Solver

EE2211 Solver is a small Python toolkit for working through common EE2211-style
linear algebra, regression, classification, correlation, gradient descent, and
decision-tree calculations. It provides both a Streamlit WebUI and an
interactive terminal menu, so you can use it either visually or from the command
line.

## Features

- Parse and inspect numeric arrays, including rank, transpose, inverse, and
  bias-column helpers.
- Solve linear systems in both `Xw = y` and `wX = y` forms.
- Fit linear, polynomial, ridge, and ridge-polynomial regression models.
- Run one-hot linear and polynomial classification workflows.
- Calculate Pearson correlation for one or more input/output columns.
- Step through symbolic gradient descent expressions with SymPy.
- Compute classification-tree impurity using Gini, entropy, or classification
  error.
- Evaluate one-level regression-tree MSE splits and find the best threshold.
- Cache recent matrices/results in the WebUI for reuse and copy-paste.

## Requirements

- Python 3.13 or newer
- [uv](https://docs.astral.sh/uv/) for dependency management

Project dependencies are declared in [pyproject.toml](pyproject.toml).

## Setup

Install dependencies with uv:

```bash
uv sync
```

## Run The WebUI

Start the Streamlit interface:

```bash
uv run streamlit run webui.py
```

The WebUI includes tabs for arrays, linear-system solving, regression and
classification, Pearson correlation, gradient descent, decision trees, and a
matrix cache. Streamlit will print a local URL in the terminal after startup.

The WebUI stores persisted widget state and cached matrices under `temp/`.
That directory is intentionally ignored by Git.

## Run The CLI

Start the interactive terminal menu:

```bash
uv run python main.py
```

The CLI accepts either menu numbers or shortcuts such as `linear`, `ridge`,
`pearson`, `gradient`, `impurity`, and `split`.

## Input Format

Most matrix prompts accept one-line arrays in any of these forms:

```text
1 2; 3 4
1, 2; 3, 4
[[1, 2], [3, 4]]
```

In the WebUI, matrix text areas also accept newline-separated rows:

```text
1 2
3 4
```

For the CLI, rows should stay on one input line and be separated with
semicolons. The CLI also supports cache shortcuts:

- `-` reuses the last input array.
- `_` reuses the last result array.

## Run Tests

Run the project test suite with:

```bash
uv run python run_tests.py
```

The tests cover the core computations used by both the CLI and WebUI.
