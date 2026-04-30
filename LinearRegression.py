import numpy as np

from CommonUtils import calculate_closed_form_weights, print_error_summary, print_system_info, print_value, print_weights


def fit_linear_regression(X, y):
    system, w = calculate_closed_form_weights(X, y)
    print_system_info("X", X, system=system)
    print_weights(w)
    print("")

    y_calculated = X @ w
    sum_of_square, mean_squared_error = print_error_summary(y, y_calculated)
    print_value("predicted y", y_calculated)

    return system, w, sum_of_square, mean_squared_error


def predict_linear_regression(w, X_test):
    y_predicted = X_test @ w
    print_value("y_predicted", y_predicted)
    print_value("y_predicted_classified", np.sign(y_predicted))
    print("")
    return y_predicted


def linear_regression(X, y, X_test=None):
    system, w, sum_of_square, mean_squared_error = fit_linear_regression(X, y)
    y_predicted = None
    if X_test is not None:
        y_predicted = predict_linear_regression(w, X_test)
    return (system, w, sum_of_square, mean_squared_error, y_predicted)
