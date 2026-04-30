import ast
import os
from traceback import print_exc

import numpy as np

from CommonUtils import (
	add_bias_column,
	ensure_2d,
	format_number,
	pretty_print_array,
	validate_same_feature_count,
	validate_same_sample_count,
)
from DecisionTreeUtils import (
	print_classification_impurity_summary,
	print_regression_split_candidates,
	print_regression_threshold_summary,
	regression_threshold_summary,
	find_best_regression_split,
)
from GradientDescent import gradient_descent, infer_variables, print_gradient_descent_result

from LinearRegression import linear_regression
from OneHotLinearClassification import (
	fit_onehot_linearclassification,
	fit_onehot_polynomialclassification,
	predict_onehot_linearclassification,
	predict_onehot_polynomialclassification,
)
from PolynomialRegression import fit_polynomial_regression, predict_polynomial_regression
from RidgePolynomialRegression import fit_ridge_poly_regression, predict_ridge_poly_regression
from RidgeRegression import fit_ridge_regression, predict_ridge_regression
from pearson_correlation import pearson_correlation


MENU_TEXT = """EE2211 Solver
| 0. Exit                                                |
| 1. Parse and inspect array                             |
| 2. Solve Xw = y                                        |
| 3. Solve wX = y                                        |
| 4. Linear regression                                   |
| 5. Polynomial regression                               |
| 6. Ridge regression                                    |
| 7. Ridge polynomial regression                         |
| 8. One-hot linear classification                       |
| 9. One-hot polynomial classification                   |
| 10. Pearson correlation                                |
| 11. Show cached input/result                           |
| 12. Gradient descent                                   |
| 13. Classification tree impurity                       |
| 14. Regression tree MSE/split                          |
+--------------------------------------------------------+"""

INPUT_HELP = (
	"Input format: exactly one line. Separate rows with ';', e.g. 1 2; 3 4.\n"
	"You may also paste one-line Python-style arrays such as [[1, 2], [3, 4]].\n"
	"Shortcuts: '-' uses the last input array, '_' uses the last result array."
)

LAST_INPUT = None
LAST_RESULT = None
VERBOSE = False


def clear():
	os.system("cls" if os.name == "nt" else "clear")


def print_section(title):
	print(f"\n{title}\n{'=' * len(title)}")

def parse_numeric_array(raw_text):
	text = raw_text.strip()
	if not text:
		raise ValueError("Input is empty.")
	if "\n" in text or "\r" in text:
		raise ValueError("New lines are not supported. Separate rows with ';' or use a one-line Python array.")

	if text.startswith("[") or text.startswith("("):
		parsed = ast.literal_eval(text)
		return ensure_2d(parsed)

	rows = []
	for line in text.replace(";", "\n").splitlines():
		stripped = line.strip()
		if not stripped:
			continue
		if "," in stripped:
			parts = [part.strip() for part in stripped.split(",") if part.strip()]
		else:
			parts = [part for part in stripped.split() if part]
		rows.append([float(part) for part in parts])

	if not rows:
		raise ValueError("No numeric rows were found.")

	widths = {len(row) for row in rows}
	if len(widths) != 1:
		raise ValueError("All rows must have the same number of values.")

	return ensure_2d(rows)


def parse_mixed_array(raw_text):
	text = raw_text.strip()
	if not text:
		raise ValueError("Input is empty.")
	if "\n" in text or "\r" in text:
		raise ValueError("New lines are not supported. Separate rows with ';' or use a one-line Python array.")

	try:
		return parse_numeric_array(raw_text)
	except (SyntaxError, ValueError):
		if text.startswith("[") or text.startswith("("):
			parsed = ast.literal_eval(text)
			arr = np.asarray(parsed, dtype=object)
			if arr.ndim == 0:
				return arr.reshape(1, 1)
			if arr.ndim == 1:
				return arr.reshape(-1, 1)
			return arr

		rows = []
		for line in text.replace(";", "\n").splitlines():
			stripped = line.strip()
			if not stripped:
				continue
			if "," in stripped:
				parts = [part.strip() for part in stripped.split(",") if part.strip()]
			else:
				parts = [stripped]
			rows.append(parts)

		if not rows:
			raise ValueError("No rows were found.")

		widths = {len(row) for row in rows}
		if len(widths) != 1:
			raise ValueError("All rows must have the same number of values.")

		return np.asarray(rows, dtype=object)


def cache_input(array):
	global LAST_INPUT
	LAST_INPUT = ensure_2d(array)
	return LAST_INPUT


def cache_result(array):
	global LAST_RESULT
	LAST_RESULT = ensure_2d(array)
	return LAST_RESULT


def prompt_float(prompt, default=None):
	while True:
		suffix = f" [{default}]" if default is not None else ""
		raw = input(f"{prompt}{suffix}: ").strip()
		if not raw and default is not None:
			return float(default)
		try:
			return float(raw)
		except ValueError:
			print("Please enter a numeric value.")


def prompt_int(prompt, default=None, minimum=None):
	while True:
		suffix = f" [{default}]" if default is not None else ""
		raw = input(f"{prompt}{suffix}: ").strip()
		if not raw and default is not None:
			value = int(default)
		else:
			try:
				value = int(raw)
			except ValueError:
				print("Please enter an integer.")
				continue
		if minimum is not None and value < minimum:
			print(f"Please enter an integer greater than or equal to {minimum}.")
			continue
		return value


def prompt_yes_no(prompt, default=True):
	default_text = "Y/n" if default else "y/N"
	while True:
		raw = input(f"{prompt} [{default_text}]: ").strip().lower()
		if not raw:
			return default
		if raw in {"y", "yes"}:
			return True
		if raw in {"n", "no"}:
			return False
		print("Please enter y or n.")


def prompt_choice(prompt, options, default=None):
	normalized = {option.lower(): option for option in options}
	options_text = "/".join(options)
	while True:
		suffix = f" [{default}]" if default is not None else ""
		raw = input(f"{prompt} ({options_text}){suffix}: ").strip().lower()
		if not raw and default is not None:
			return default
		if raw in normalized:
			return normalized[raw]
		print(f"Please choose one of: {options_text}.")


def prompt_optional_float(prompt):
	raw = input(f"{prompt} [skip]: ").strip()
	if not raw:
		return None
	try:
		return float(raw)
	except ValueError:
		print("Please enter a numeric value or leave blank to skip.")
		return prompt_optional_float(prompt)


def prompt_gradient_cost_expression():
	while True:
		cost_expression = input("Cost function C(...): ").strip()
		if not cost_expression:
			print("Cost function is required.")
			continue
		try:
			variables = infer_variables(cost_expression)
		except Exception as exc:
			print(f"Input error: {exc}")
			continue
		if not variables:
			print("Cost function must contain at least one variable.")
			continue
		return cost_expression, variables


def prompt_gradient_initial_values(variables):
	variable_text = ", ".join(variables)
	while True:
		raw = input(f"Initial values for {variable_text} (comma-separated): ").strip().replace("−", "-")
		if not raw:
			print("Initial values are required.")
			continue

		parts = [part.strip() for part in raw.split(",")]
		if any(not part for part in parts):
			print("Please enter one numeric value per variable, separated by commas.")
			continue
		if len(parts) != len(variables):
			print(f"Expected {len(variables)} initial values for {variable_text}, got {len(parts)}.")
			continue

		try:
			return np.asarray([float(part) for part in parts], dtype=float)
		except ValueError:
			print("Please enter numeric initial values separated by commas.")


def input_array(name, allow_last_result=True):
	print_section(f"Input {name}")
	if VERBOSE:
		print(INPUT_HELP)
	line = input("> ").strip().replace("âˆ’", "-")
	if not line:
		print("Enter one line of input.")
		return input_array(name, allow_last_result=allow_last_result)

	if line == "-":
		if LAST_INPUT is None:
			print("No cached input is available.")
			return input_array(name, allow_last_result=allow_last_result)
		return cache_input(LAST_INPUT)

	if line == "_" and allow_last_result:
		if LAST_RESULT is None:
			print("No cached result is available.")
			return input_array(name, allow_last_result=allow_last_result)
		return cache_input(LAST_RESULT)

	try:
		return cache_input(parse_numeric_array(line))
	except (SyntaxError, ValueError) as exc:
		print(f"Input error: {exc}")
		return input_array(name, allow_last_result=allow_last_result)


def input_optional_array(name, allow_last_result=True):
	print_section(f"Input {name}")
	if VERBOSE:
		print(INPUT_HELP)
	print("Leave this input empty to skip it and return to the main menu after fitting.")
	line = input("> ").strip().replace("âˆ’", "-")
	if not line:
		return None

	if line == "-":
		if LAST_INPUT is None:
			print("No cached input is available.")
			return input_optional_array(name, allow_last_result=allow_last_result)
		return cache_input(LAST_INPUT)

	if line == "_" and allow_last_result:
		if LAST_RESULT is None:
			print("No cached result is available.")
			return input_optional_array(name, allow_last_result=allow_last_result)
		return cache_input(LAST_RESULT)

	try:
		return cache_input(parse_numeric_array(line))
	except (SyntaxError, ValueError) as exc:
		print(f"Input error: {exc}")
		return input_optional_array(name, allow_last_result=allow_last_result)


def input_target_array(name):
	print_section(f"Input {name}")
	if VERBOSE:
		print(INPUT_HELP)
	print("For classification targets, you may enter one-hot rows or labels such as class1;class2;class3.")
	line = input("> ").strip().replace("âˆ’", "-")
	if not line:
		print("Enter one line of input.")
		return input_target_array(name)

	try:
		return parse_mixed_array(line)
	except (SyntaxError, ValueError) as exc:
		print(f"Input error: {exc}")
		return input_target_array(name)


def system_shape_description(matrix):
	rows, cols = matrix.shape
	if rows > cols:
		return "overdetermined"
	if rows < cols:
		return "underdetermined"
	return "evendetermined"


def print_solve_result(title, coefficient_matrix, target_matrix, result_name, orient_result=lambda value: value):
	rows, variables = coefficient_matrix.shape
	system_shape = system_shape_description(coefficient_matrix)
	rank_coefficients = np.linalg.matrix_rank(coefficient_matrix)
	rank_augmented = np.linalg.matrix_rank(np.hstack((coefficient_matrix, target_matrix)))
	is_consistent = rank_coefficients == rank_augmented

	print_section(title)
	print(f"system: {system_shape} ({rows} equations, {variables} unknowns)")
	print(f"rank(coefficients): {rank_coefficients}")
	print(f"rank(augmented): {rank_augmented}")

	solution, _, _, _ = np.linalg.lstsq(coefficient_matrix, target_matrix, rcond=None)
	display_solution = orient_result(solution)
	predicted_target = coefficient_matrix @ solution
	residual = predicted_target - target_matrix
	mean_squared_error = np.mean(np.square(residual), axis=0, keepdims=True)
	standard_error = np.sqrt(mean_squared_error)

	if is_consistent:
		if rank_coefficients == variables:
			print("solution: unique exact solution")
		else:
			print("solution: infinitely many exact solutions; showing one minimum-norm solution")
		pretty_print_array(result_name, display_solution, show_python=False)
		pretty_print_array("standard error", orient_result(standard_error), show_python=False)
		pretty_print_array("mse", orient_result(mean_squared_error), show_python=False)
		cache_result(display_solution)
		return

	print("solution: no exact solution")
	pretty_print_array(f"{result_name}_hat", display_solution, show_python=False)
	pretty_print_array("residual", orient_result(residual), show_python=False)
	pretty_print_array("standard error", orient_result(standard_error), show_python=False)
	pretty_print_array("mse", orient_result(mean_squared_error), show_python=False)
	cache_result(display_solution)


def inspect_array_details(array):
	rows, cols = array.shape
	rank = np.linalg.matrix_rank(array)
	print(f"shape: {rows} x {cols}")
	print(f"rank: {rank}")

	is_square = rows == cols
	if is_square:
		determinant = np.linalg.det(array)
		print("determinant exists: yes")
		print(f"determinant: {format_number(determinant)}")
	else:
		print("determinant exists: no")

	inverse_exists = is_square and rank == rows
	left_inverse_exists = rank == cols
	right_inverse_exists = rank == rows
	print(f"inverse exists: {'yes' if inverse_exists else 'no'}")
	print(f"left inverse exists: {'yes' if left_inverse_exists else 'no'}")
	print(f"right inverse exists: {'yes' if right_inverse_exists else 'no'}")

	if inverse_exists:
		pretty_print_array("inverse", np.linalg.inv(array), show_python=False)
	if left_inverse_exists:
		left_inverse = np.linalg.inv(array.T @ array) @ array.T
		pretty_print_array("left_inverse", left_inverse, show_python=False)
	if right_inverse_exists:
		right_inverse = array.T @ np.linalg.inv(array @ array.T)
		pretty_print_array("right_inverse", right_inverse, show_python=False)


def inspect_array():
	array = input_array("array")
	cache_result(array)
	pretty_print_array("array", array)
	inspect_array_details(array)


def run_solve():
	X = input_array("X")
	y = input_array("y")
	validate_same_sample_count(X, y)
	print_solve_result("Solve Xw = y", X, y, "w")


def run_left_solve():
	X = input_array("X")
	y = input_array("y")
	if X.shape[1] != y.shape[1]:
		raise ValueError(
			f"Column count mismatch: X has {X.shape[1]} columns but y has {y.shape[1]} columns."
		)

	print_solve_result("Solve wX = y", X.T, y.T, "w", orient_result=lambda value: value.T)


def run_linear_regression():
	X = input_array("X")
	Y = input_array("Y")
	validate_same_sample_count(X, Y)

	include_bias = prompt_yes_no("Add bias/offset term", default=True)
	X_model = add_bias_column(X) if include_bias else X

	_, w, _, _, _ = linear_regression(X_model, Y)
	cache_result(w)

	X_test = input_optional_array("X_test")
	if X_test is None:
		return
	validate_same_feature_count(X, X_test)
	X_test_model = add_bias_column(X_test) if include_bias else X_test
	y_predicted = X_test_model @ w
	pretty_print_array("y_predicted", y_predicted, show_python=False, show_rank=False)
	cache_result(y_predicted)


def run_polynomial_regression():
	X = input_array("X")
	Y = input_array("Y")
	validate_same_sample_count(X, Y)

	order = prompt_int("Polynomial order", default=2, minimum=1)
	print("Polynomial regression already includes the constant term. Do not add a manual offset column.")
	poly, _, w, _, _ = fit_polynomial_regression(X, Y, order=order)
	cache_result(w)

	X_test = input_optional_array("X_test")
	if X_test is None:
		return
	validate_same_feature_count(X, X_test)
	y_predicted = predict_polynomial_regression(poly, w, X_test)
	cache_result(y_predicted)


def run_ridge_regression():
	X = input_array("X")
	Y = input_array("Y")
	validate_same_sample_count(X, Y)

	ridge_lambda = prompt_float("Lambda", default=1.0)
	form = prompt_choice("Ridge form", ["auto", "primal form", "dual form"], default="auto")
	X_model = add_bias_column(X)

	_, _, w, _, _ = fit_ridge_regression(X_model, Y, LAMBDA=ridge_lambda, form=form)
	cache_result(w)

	X_test = input_optional_array("X_test")
	if X_test is None:
		return
	validate_same_feature_count(X, X_test)
	X_test_model = add_bias_column(X_test)
	y_predicted = predict_ridge_regression(w, X_test_model)
	cache_result(y_predicted)


def run_ridge_polynomial_regression():
	X = input_array("X")
	Y = input_array("Y")
	validate_same_sample_count(X, Y)

	ridge_lambda = prompt_float("Lambda", default=1.0)
	order = prompt_int("Polynomial order", default=2, minimum=1)
	form = prompt_choice("Ridge form", ["auto", "primal form", "dual form"], default="auto")
	print("Polynomial regression already includes the constant term. Do not add a manual offset column.")
	poly, _, _, w, _, _ = fit_ridge_poly_regression(X, Y, LAMBDA=ridge_lambda, order=order, form=form)
	cache_result(w)

	X_test = input_optional_array("X_test")
	if X_test is None:
		return
	validate_same_feature_count(X, X_test)
	y_predicted = predict_ridge_poly_regression(poly, w, X_test)
	cache_result(y_predicted)


def run_onehot_linearclassification():
	X = input_array("X")
	Y = input_target_array("Y (one-hot or class labels)")
	validate_same_sample_count(X, Y)
	X_model = add_bias_column(X)

	_, w, _, class_labels, _, _ = fit_onehot_linearclassification(X_model, Y)
	cache_result(w)

	X_test = input_optional_array("X_test")
	if X_test is None:
		return
	validate_same_feature_count(X, X_test)
	X_test_model = add_bias_column(X_test)
	y_predicted = predict_onehot_linearclassification(w, X_test_model, class_labels=class_labels)
	cache_result(y_predicted)


def run_onehot_polynomial_classification():
	X = input_array("X")
	Y = input_target_array("Y (one-hot or class labels)")
	validate_same_sample_count(X, Y)

	order = prompt_int("Polynomial order", default=2, minimum=1)
	print("Polynomial classification already includes the constant term. Do not add a manual offset column.")
	poly, _, w, _, class_labels, _, _ = fit_onehot_polynomialclassification(X, Y, order=order)
	cache_result(w)

	X_test = input_optional_array("X_test")
	if X_test is None:
		return
	validate_same_feature_count(X, X_test)
	y_predicted = predict_onehot_polynomialclassification(poly, w, X_test, class_labels=class_labels)
	cache_result(y_predicted)


def run_pearson_correlation():
	X = input_array("X")
	Y = input_array("Y")
	if X.shape[1] != Y.shape[1]:
		raise ValueError(
			f"Pearson correlation expects matching observation counts: X has {X.shape[1]}, Y has {Y.shape[1]}."
		)
	cache_result(pearson_correlation(X, Y))


def run_gradient_descent():
	print_section("Gradient Descent")
	print("Use Python/SymPy syntax, e.g. x**2 + x*y**2 or sin(w)**2.")
	cost_expression, variables = prompt_gradient_cost_expression()
	print(f"Variables: {', '.join(variables)}")

	initial_values = prompt_gradient_initial_values(variables)
	learning_rate = prompt_float("Learning rate", default=0.1)
	iterations = prompt_int("Number of iterations", default=10, minimum=1)
	tolerance = prompt_optional_float("Tolerance")
	expression, gradient_expressions, history = gradient_descent(
		cost_expression,
		variables=variables,
		initial_values=initial_values,
		learning_rate=learning_rate,
		iterations=iterations,
		tolerance=tolerance,
	)
	print_gradient_descent_result(expression, variables, gradient_expressions, history)
	cache_result(history[-1]["values"].reshape(-1, 1))


def run_classification_tree_impurity():
	print_section("Classification Tree Impurity")
	print("Enter one row per node; columns are class counts. For a depth-1 overall impurity, enter child nodes only.")
	class_counts = input_array("class counts by node")
	measure = prompt_choice("Impurity measure", ["all", "gini", "entropy", "misclassification"], default="all")
	print_classification_impurity_summary(class_counts, measure=measure)


def run_regression_tree_mse():
	print_section("Regression Tree MSE/Split")
	print("Enter one-dimensional x values and matching y values. A threshold uses left: x <= threshold, right: x > threshold.")
	X = input_array("x values")
	Y = input_array("y values")
	validate_same_sample_count(X, Y)
	threshold = prompt_optional_float("Decision threshold")
	if threshold is not None:
		summary = regression_threshold_summary(X, Y, threshold)
		print_regression_threshold_summary(summary)
		cache_result([[summary["root_mse"]], [summary["overall_mse"]]])

	if prompt_yes_no("Find best one-level regression-tree split", default=threshold is None):
		best, summaries = find_best_regression_split(X, Y)
		print_regression_split_candidates(summaries)
		print("Best split:")
		print_regression_threshold_summary(best)
		cache_result([[best["threshold"]], [best["overall_mse"]]])


def show_cache():
	print_section("Cached Arrays")
	if LAST_INPUT is None:
		print("Last input: [empty]")
	else:
		pretty_print_array("last_input", LAST_INPUT)

	if LAST_RESULT is None:
		print("Last result: [empty]")
	else:
		pretty_print_array("last_result", LAST_RESULT)


def process_input():
	print(MENU_TEXT)
	option = input("Choose an option: ").strip().lower()

	actions = {
		"1": inspect_array,
		"inspect": inspect_array,
		"2": run_solve,
		"solve": run_solve,
		"xw=y": run_solve,
		"3": run_left_solve,
		"left solve": run_left_solve,
		"wx=y": run_left_solve,
		"wX=y": run_left_solve,
		"4": run_linear_regression,
		"linear": run_linear_regression,
		"lin": run_linear_regression,
		"5": run_polynomial_regression,
		"poly": run_polynomial_regression,
		"polynomial": run_polynomial_regression,
		"6": run_ridge_regression,
		"ridge": run_ridge_regression,
		"7": run_ridge_polynomial_regression,
		"ridgepoly": run_ridge_polynomial_regression,
		"ridge-polynomial": run_ridge_polynomial_regression,
		"8": run_onehot_linearclassification,
		"onehot": run_onehot_linearclassification,
		"classification": run_onehot_linearclassification,
		"9": run_onehot_polynomial_classification,
		"onehotpoly": run_onehot_polynomial_classification,
		"polyclass": run_onehot_polynomial_classification,
		"10": run_pearson_correlation,
		"pearson": run_pearson_correlation,
		"corr": run_pearson_correlation,
		"11": show_cache,
		"cache": show_cache,
		"12": run_gradient_descent,
		"gd": run_gradient_descent,
		"gradient": run_gradient_descent,
		"gradient descent": run_gradient_descent,
		"13": run_classification_tree_impurity,
		"gini": run_classification_tree_impurity,
		"entropy": run_classification_tree_impurity,
		"impurity": run_classification_tree_impurity,
		"14": run_regression_tree_mse,
		"tree mse": run_regression_tree_mse,
		"regression tree": run_regression_tree_mse,
		"split": run_regression_tree_mse,
	}

	if option in {"0", "exit", "quit", "q"}:
		return False

	action = actions.get(option)
	if action is None:
		print("Unknown option.")
	else:
		try:
			action()
		except KeyboardInterrupt:
			print("\nReturning to main menu.")
			clear()
			return True

	try:
		input("\nPress Enter to continue...")
	except KeyboardInterrupt:
		print("\nReturning to main menu.")
	clear()
	return True


def main():
	clear()
	while True:
		try:
			if not process_input():
				return
		except KeyboardInterrupt:
			print("\nCtrl+C detected, exiting.")
			return
		except Exception as exc:
			print(f"\nERROR: {exc}")
			print_exc()
			input("\nPress Enter to continue...")
			clear()


if __name__ == "__main__":
	main()
