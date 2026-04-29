import contextlib
import io
import unittest

import numpy as np

from CommonUtils import add_bias_column, calculate_closed_form_weights, calculate_ridge_weights
from DecisionTreeUtils import classification_impurity_summary, find_best_regression_split, regression_threshold_summary
from GradientDescent import gradient_descent
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


@contextlib.contextmanager
def quiet_stdout():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


class SolverComputationTests(unittest.TestCase):
    def test_closed_form_and_linear_regression(self):
        X = np.array([[1, 0], [1, 1], [1, 2]], dtype=float)
        y = np.array([[1], [3], [5]], dtype=float)

        system, expected_w = calculate_closed_form_weights(X, y)
        self.assertEqual(system, "overdetermined")
        np.testing.assert_allclose(expected_w, [[1], [2]], atol=1e-10)

        with quiet_stdout():
            _, w, _, _, y_predicted = linear_regression(X, y, np.array([[1, 3]], dtype=float))
        np.testing.assert_allclose(w, expected_w, atol=1e-10)
        np.testing.assert_allclose(y_predicted, [[7]], atol=1e-10)

    def test_polynomial_regression_prediction(self):
        X = np.array([[0], [1], [2]], dtype=float)
        y = 1 + 2 * X + X**2
        with quiet_stdout():
            poly, _, w, _, _ = fit_polynomial_regression(X, y, order=2)
            y_predicted = predict_polynomial_regression(poly, w, np.array([[3]], dtype=float))
        np.testing.assert_allclose(w.reshape(-1), [1, 2, 1], atol=1e-10)
        np.testing.assert_allclose(y_predicted, [[16]], atol=1e-10)

    def test_ridge_regression(self):
        X = np.array([[1, 0], [1, 1], [1, 2]], dtype=float)
        y = np.array([[1], [3], [5]], dtype=float)
        ridge_lambda = 0.5
        _, _, expected_w = calculate_ridge_weights(X, y, ridge_lambda, form="primal form")

        with quiet_stdout():
            _, _, w, _, _ = fit_ridge_regression(X, y, LAMBDA=ridge_lambda, form="primal form")
            y_predicted = predict_ridge_regression(w, np.array([[1, 3]], dtype=float))
        np.testing.assert_allclose(w, expected_w, atol=1e-10)
        np.testing.assert_allclose(y_predicted, np.array([[1, 3]], dtype=float) @ expected_w, atol=1e-10)

    def test_ridge_polynomial_regression_prediction(self):
        X = np.array([[0], [1], [2], [3]], dtype=float)
        y = 1 + X
        with quiet_stdout():
            poly, _, _, w, _, _ = fit_ridge_poly_regression(X, y, LAMBDA=0.1, order=2, form="primal form")
            y_predicted = predict_ridge_poly_regression(poly, w, np.array([[4]], dtype=float))
        expected = poly.transform(np.array([[4]], dtype=float)) @ w
        np.testing.assert_allclose(y_predicted, expected, atol=1e-10)

    def test_onehot_classification(self):
        X = add_bias_column(np.array([[0], [1], [2], [3]], dtype=float))
        y = np.array([["low"], ["low"], ["high"], ["high"]], dtype=object)

        with quiet_stdout():
            _, w, _, labels, _, _ = fit_onehot_linearclassification(X, y)
            y_predicted = predict_onehot_linearclassification(w, np.array([[1, 2.5]], dtype=float), labels)
        self.assertEqual(labels, ["low", "high"])
        self.assertEqual(labels[int(np.argmax(y_predicted[0]))], "high")

    def test_onehot_polynomial_classification_prediction(self):
        X = np.array([[0], [1], [2], [3]], dtype=float)
        y = np.array([["left"], ["left"], ["right"], ["right"]], dtype=object)
        with quiet_stdout():
            poly, _, w, _, labels, _, _ = fit_onehot_polynomialclassification(X, y, order=2)
            y_predicted = predict_onehot_polynomialclassification(poly, w, np.array([[2.5]], dtype=float), labels)
        expected = poly.transform(np.array([[2.5]], dtype=float)) @ w
        np.testing.assert_allclose(y_predicted, expected, atol=1e-10)

    def test_pearson_correlation_for_non_six_observations(self):
        X = np.array([[1, 2, 3, 4], [1, 1, 2, 2]], dtype=float)
        Y = np.array([[2, 4, 6, 8]], dtype=float)
        with quiet_stdout():
            pearson = pearson_correlation(X, Y)
        np.testing.assert_allclose(pearson[0, 0], 1.0, atol=1e-10)
        np.testing.assert_allclose(pearson[1, 0], 1 / np.sqrt(1.25), atol=1e-10)

    def test_gradient_descent_lecture_and_past_paper_examples(self):
        _, _, history = gradient_descent("x**2", variables=["x"], initial_values=[1], learning_rate=0.4, iterations=4)
        np.testing.assert_allclose(history[-1]["values"], [0.0016], atol=1e-12)

        _, _, history = gradient_descent("sin(w)**2", variables=["w"], initial_values=[3], learning_rate=0.1, iterations=1)
        np.testing.assert_allclose(history[1]["gradient"], [2 * np.sin(3) * np.cos(3)], atol=1e-12)
        np.testing.assert_allclose(history[1]["values"], [3.0279415498198925], atol=1e-12)

        _, _, history = gradient_descent("x**2 + x*y**2", variables=["x", "y"], initial_values=[3, 2], learning_rate=0.2, iterations=1)
        np.testing.assert_allclose(history[1]["gradient"], [10, 12], atol=1e-12)
        np.testing.assert_allclose(history[1]["values"], [1, -0.4], atol=1e-12)

    def test_classification_tree_impurity(self):
        _, impurities, overall = classification_impurity_summary([[2, 5], [6, 0]], measure="gini")
        np.testing.assert_allclose(impurities, [20 / 49, 0], atol=1e-10)
        np.testing.assert_allclose(overall, 7 / 13 * 20 / 49, atol=1e-10)

        _, impurities, overall = classification_impurity_summary([[2, 5], [6, 0]], measure="entropy")
        expected_entropy = -(2 / 7) * np.log2(2 / 7) - (5 / 7) * np.log2(5 / 7)
        np.testing.assert_allclose(impurities, [expected_entropy, 0], atol=1e-10)
        np.testing.assert_allclose(overall, 7 / 13 * expected_entropy, atol=1e-10)

    def test_regression_tree_mse_and_best_split(self):
        x = np.array([0.2, 0.7, 1.8, 2.2, 3.7, 4.1, 4.5, 5.1, 6.3, 7.4])
        y = np.array([2.1, 1.5, 5.8, 6.1, 9.1, 9.5, 9.8, 12.7, 13.8, 15.9])
        summary = regression_threshold_summary(x, y, 3)
        np.testing.assert_allclose(summary["root_mse"], 20.6381, atol=1e-4)
        np.testing.assert_allclose(summary["overall_mse"], 5.5648, atol=1e-4)

        best, summaries = find_best_regression_split(x, y)
        self.assertEqual(len(summaries), len(np.unique(x)) - 1)
        self.assertLessEqual(best["overall_mse"], summary["overall_mse"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
