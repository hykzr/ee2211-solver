import numpy as np
from sklearn.preprocessing import PolynomialFeatures

from CommonUtils import calculate_ridge_weights, print_error_summary, print_system_info, print_value, print_weights, pretty_print_array


def fit_ridge_poly_regression(X, y, LAMBDA, order, form):
    poly = PolynomialFeatures(order)
    P = poly.fit_transform(X)
    print("the number of parameters: ", P.shape[1])
    print("the number of samples: ", P.shape[0])
    system, resolved_form, w = calculate_ridge_weights(P, y, ridge_lambda=LAMBDA, form=form)

    print_system_info("P", P, system=system, form=resolved_form)
    pretty_print_array("the polynomial transformed matrix P", P)
    print("")
    print_weights(w)
    print("")

    P_train_predicted = P @ w
    print_value("y_train_predicted", np.sign(P_train_predicted))
    sum_of_square, mean_squared_error = print_error_summary(y, P_train_predicted)

    return poly, system, resolved_form, w, sum_of_square, mean_squared_error


def predict_ridge_poly_regression(poly, w, X_test):
    P_test = poly.fit_transform(X_test)
    pretty_print_array("transformed test sample P_test", P_test)
    print("")
    y_predicted = P_test @ w
    print_value("y_predicted", y_predicted)
    return y_predicted


def ridge_poly_regression(X, y, LAMBDA, order, form, X_test=None):
    poly, system, resolved_form, w, sum_of_square, mean_squared_error = fit_ridge_poly_regression(
        X, y, LAMBDA=LAMBDA, order=order, form=form
    )
    y_predicted = None
    if X_test is not None:
        y_predicted = predict_ridge_poly_regression(poly, w, X_test)
    return (system, resolved_form, w, sum_of_square, mean_squared_error, y_predicted)

    # if single class classification
    # y_classified = np.sign(y_predicted)
    # print("y_classified is", y_classified)
    #HI
    # return(system, P, w, y_predicted, y_classified))










def ridge_poly_regression_simplified(X, y, LAMBDA, order, form, X_test, y_test):
    poly = PolynomialFeatures(order)
    P = poly.fit_transform(X)
    _, _, w = calculate_ridge_weights(P, y, ridge_lambda=LAMBDA, form=form)

    P_test = poly.fit_transform(X_test)

    P_train_predicted = P @ w
    _, mean_squared_error = print_error_summary(y, P_train_predicted, prefix="ridge train")

    P_test_predicted = P_test @ w
    _, mean_squared_error = print_error_summary(y_test, P_test_predicted, prefix="ridge test")

