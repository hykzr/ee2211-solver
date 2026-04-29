import numpy as np
from sklearn.preprocessing import PolynomialFeatures

from CommonUtils import calculate_closed_form_weights, print_error_summary, print_system_info, print_value, print_weights, pretty_print_array


def fit_polynomial_regression(X, y, order):
    poly = PolynomialFeatures(order)
    P = poly.fit_transform(X)
    print("the number of parameters: ", P.shape[1])
    print("the number of samples: ", P.shape[0])
    system, w = calculate_closed_form_weights(P, y)
    print_system_info("P", P, system=system)
    pretty_print_array("the polynomial transformed matrix P", P)
    print("")
    print_weights(w)
    print("")

    P_train_predicted = P @ w
    print_value("y_train_predicted", P_train_predicted)
    print_value("y_train_classified", np.sign(P_train_predicted))
    sum_of_square, mean_squared_error = print_error_summary(y, P_train_predicted)

    return poly, system, w, sum_of_square, mean_squared_error


def predict_polynomial_regression(poly, w, X_test):
    P_test = poly.transform(X_test)
    pretty_print_array("transformed test sample P_test", P_test)
    print("")
    y_predicted = P_test @ w
    print_value("y_predicted", y_predicted)
    return y_predicted


def polynomial_regression(X, y, order, X_test=None):
    poly, system, w, sum_of_square, mean_squared_error = fit_polynomial_regression(X, y, order)
    y_predicted = None
    if X_test is not None:
        y_predicted = predict_polynomial_regression(poly, w, X_test)
    return (system, w, sum_of_square, mean_squared_error, y_predicted)


    # if single class classification
    # y_classified = np.sign(y_predicted)
    # print("y_classified is", y_classified)
    #
    # return(system, P, w, y_predicted, y_classified)

    # print("P rank:", np.linalg.matrix_rank(P))
    # result=np.hstack((P,y))
    # print("P|y rank: ", np.linalg.matrix_rank(result))
