from CommonUtils import calculate_ridge_weights, print_error_summary, print_system_info, print_value, print_weights


def fit_ridge_regression(X, y, LAMBDA, form="auto"):
    system, resolved_form, w = calculate_ridge_weights(X, y, ridge_lambda=LAMBDA, form=form)
    print_system_info("X", X, system=system, form=resolved_form)
    print_weights(w)
    print("")

    y_calculated = X @ w
    print_value("y_calculated", y_calculated)
    sum_of_square, mean_squared_error = print_error_summary(y, y_calculated)

    return system, resolved_form, w, sum_of_square, mean_squared_error


def predict_ridge_regression(w, X_test):
    y_predicted = X_test @ w
    print_value("y_predicted", y_predicted)
    return y_predicted


def ridge_regression(X, y, LAMBDA, X_test=None, form="auto"):
    system, resolved_form, w, sum_of_square, mean_squared_error = fit_ridge_regression(
        X, y, LAMBDA=LAMBDA, form=form
    )
    y_predicted = None
    if X_test is not None:
        y_predicted = predict_ridge_regression(w, X_test)
    return (system, resolved_form, w, sum_of_square, mean_squared_error, y_predicted)
