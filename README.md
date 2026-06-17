# EE2211 Solver

![EE2211 Solver cover](images/cover.png)

EE2211 Solver is a small helper app I made for working through EE2211-style
machine learning calculation questions faster.

It is not meant to replace understanding the concepts. The goal is more like:
let the app handle repeated matrix / regression / gradient / tree / clustering
calculation steps, so you can spend more exam time thinking about what the
question is actually asking.

The main way to use it is the Streamlit WebUI. There is also a terminal menu if
you prefer command line tools, but beginners should start with the WebUI.

## What It Can Do

- Parse matrices and quickly check shape, rank, transpose, inverse, left inverse,
  and right inverse.
- Solve linear systems such as `Xw = y` and `wX = y`.
- Fit linear, polynomial, ridge, and ridge-polynomial regression models.
- Run one-hot linear / polynomial classification workflows.
- Calculate Pearson correlation for one or more columns.
- Step through symbolic gradient descent with iteration history.
- Compute decision-tree impurity with Gini, entropy, and classification error.
- Check one-level regression-tree splits and find the best threshold.
- Run K-means clustering from points and initial centroids.
- Cache matrices and results in the WebUI so you can copy and reuse them.

## Beginner Setup

You do not need to be good at Python to run this. You mainly need to install one
tool called `uv`, then copy-paste a few commands.

### 0. Install python

### 1. Install `uv`

If you already have `uv`, skip this step.

[Installing uv](https://docs.astral.sh/uv/getting-started/installation/)

### 2. Clone This Repo

```bash
git clone https://github.com/hykzr/demo-video-recorder.git
cd demo-video-recorder
```

### 3. Install The App Dependencies

Run:

```bash
uv sync
```

This may take a little while the first time. It installs the Python packages the
solver needs.

### 4. Start The WebUI

Run:

```bash
uv run streamlit run webui.py
```

Streamlit will print a local link, usually something like:

```text
http://localhost:8501
```

Open that link in your browser. That is the solver.

## How To Type Matrices

Most input boxes accept simple rows like this:

```text
1 2
3 4
```

You can also use semicolons:

```text
1 2; 3 4
```

Or Python-style arrays:

```text
[[1, 2], [3, 4]]
```

For long one-dimensional data, a useful trick is to type it sideways first:

```text
0 1 2 3 4 5
```

Then click the `Transpose` button to turn it into a column.

## Cache Tip

The WebUI saves recent inputs and results in the `Cache` tab. This is useful
when you want to copy a model matrix, a weight vector, predictions, centroids,
or any intermediate result and paste it into another tab.

The app stores temporary WebUI state and cached matrices under `temp/`. That
folder is ignored by Git.

## Optional: Terminal Version

If you prefer a text menu instead of the browser app, run:

```bash
uv run python main.py
```

You can choose menu items by number, or use shortcuts like `linear`, `ridge`,
`pearson`, `gradient`, `impurity`, and `split`.

## Run Tests

If you want to check that the solver logic is still working:

```bash
uv run python run_tests.py
```

## Exam Rules And Academic Integrity

Please check the current EE2211 exam rules before using this in any exam.
For my year, pre-written offline code / offline materials were allowed for the
open-book Examplify exams, as long as they followed the stated restrictions.
That does **not** automatically mean the same rules apply every year. The
teaching team can change the policy, and your exam instructions are the source
of truth.

Before relying on this solver in an exam, confirm things like:

- whether pre-coded scripts are allowed;
- whether using code from others (e.g. from a public repository) is allowed.

The solver itself, although coded partially by llm, does not need internet access, nor does it use any local llm based tools.

Use the repo responsibly. It is a study and calculation helper, not permission
to ignore course or exam rules.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).

That means you can use, modify, and share it, but it is provided as-is with no
warranty. The authors are not responsible for misuse, wrong answers, rule
violations, or any consequences from using it in a setting where it is not
allowed.
