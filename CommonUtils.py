import numpy as np
import sympy as sp


def ensure_2d(array):
    arr = np.asarray(array, dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def system_type(matrix):
    rows, cols = matrix.shape
    if cols < rows:
        return "overdetermined"
    if cols > rows:
        return "underdetermined"
    return "full rank"


def format_number(value):
    value = float(value)
    if not np.isfinite(value):
        return str(value)
    if np.isclose(value, round(value), atol=1e-10):
        return str(int(round(value)))

    abs_value = abs(value)
    if abs_value != 0 and (abs_value >= 1e4 or abs_value < 1e-3):
        text = f"{value:.4g}"
    else:
        text = f"{value:.4f}".rstrip("0").rstrip(".")

    if text == "-0":
        return "0"
    return text


def _to_pretty_sympy_matrix(array):
    arr = ensure_2d(array)
    return sp.Matrix(
        [
            [sp.sympify(format_number(value), evaluate=False) for value in row]
            for row in arr.tolist()
        ]
    )


def python_array_literal(array):
    arr = ensure_2d(array)
    rows = []
    for row in arr.tolist():
        rows.append("[" + ", ".join(format_number(value) for value in row) + "]")
    return "[" + ", ".join(rows) + "]"


def pretty_print_array(name, array, show_python=True, show_rank=False):
    arr = ensure_2d(array)
    print(f"{name}:")
    if show_rank:
        print(f"rank: {np.linalg.matrix_rank(arr)}")
    sp.pprint(_to_pretty_sympy_matrix(arr), use_unicode=True)
    if show_python:
        print(f"python: {python_array_literal(arr)}")


def print_value(name, value, show_python=False, show_rank=False):
    arr = np.asarray(value)
    if arr.ndim == 0:
        print(f"{name}: {format_number(arr.item())}")
        return
    pretty_print_array(name, arr, show_python=show_python, show_rank=show_rank)


def print_label_vector(name, values):
    arr = np.asarray(values, dtype=object).reshape(-1)
    print(f"{name}:")
    for value in arr.tolist():
        print(value)


def validate_same_sample_count(X, y):
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"Sample count mismatch: X has {X.shape[0]} rows but y has {y.shape[0]} rows."
        )


def validate_same_feature_count(X, X_test):
    if X.shape[1] != X_test.shape[1]:
        raise ValueError(
            f"Feature mismatch: X has {X.shape[1]} columns but X_test has {X_test.shape[1]} columns."
        )


def add_bias_column(matrix):
    matrix = ensure_2d(matrix)
    return np.hstack((np.ones((matrix.shape[0], 1)), matrix))


def calculate_closed_form_weights(X, y):
    system = system_type(X)
    if system == "overdetermined":
        w = np.linalg.inv(X.T @ X) @ X.T @ y
    elif system == "underdetermined":
        w = X.T @ np.linalg.inv(X @ X.T) @ y
    else:
        w = np.linalg.inv(X) @ y
    return system, w


def resolve_ridge_form(X, form="auto"):
    system = system_type(X)
    if form == "auto":
        if system == "overdetermined":
            return system, "primal form"
        if system == "underdetermined":
            return system, "dual form"
        return system, "full rank"
    return system, form


def calculate_ridge_weights(X, y, ridge_lambda, form="auto"):
    system, resolved_form = resolve_ridge_form(X, form)
    if resolved_form == "primal form":
        identity = np.identity(X.shape[1])
        w = np.linalg.inv(X.T @ X + ridge_lambda * identity) @ X.T @ y
    elif resolved_form == "dual form":
        identity = np.identity(X.shape[0])
        w = X.T @ np.linalg.inv(X @ X.T + ridge_lambda * identity) @ y
    else:
        w = np.linalg.inv(X) @ y
    return system, resolved_form, w


def squared_error_summary(y_true, y_pred):
    residual = y_pred - y_true
    squared_error = np.square(residual)
    sum_of_square = np.sum(squared_error, axis=0)
    mean_squared_error = sum_of_square / y_true.shape[0]
    return sum_of_square, mean_squared_error


def print_error_summary(y_true, y_pred, prefix=""):
    sum_of_square, mean_squared_error = squared_error_summary(y_true, y_pred)
    label_prefix = f"{prefix} " if prefix else ""
    print_value(f"{label_prefix}square error", sum_of_square)
    print_value(f"{label_prefix}MEAN square error", mean_squared_error)
    print("")
    return sum_of_square, mean_squared_error


def print_weights(w):
    pretty_print_array("w", w, show_python=False)


def print_system_info(matrix_name, matrix, system=None, form=None):
    if system is None:
        system = system_type(matrix)
    # print_shape_and_rank(matrix_name, matrix)
    if form is None:
        print(system, "system\n")
    else:
        print(system, "system  ", form)
        print("")