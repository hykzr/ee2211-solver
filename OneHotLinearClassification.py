import numpy as np
from sklearn.preprocessing import PolynomialFeatures

from CommonUtils import calculate_closed_form_weights, print_error_summary, print_label_vector, print_system_info, print_value, print_weights


def _encode_label_column(labels):
    class_labels = []
    class_to_index = {}
    encoded_indices = []

    for label in labels:
        if label not in class_to_index:
            class_to_index[label] = len(class_labels)
            class_labels.append(label)
        encoded_indices.append(class_to_index[label])

    encoded = np.zeros((len(labels), len(class_labels)), dtype=float)
    encoded[np.arange(len(labels)), encoded_indices] = 1.0
    return encoded, class_labels


def prepare_class_targets(y):
    raw = np.asarray(y, dtype=object)
    if raw.ndim == 0:
        raw = raw.reshape(1, 1)
    elif raw.ndim == 1:
        raw = raw.reshape(-1, 1)

    has_text = any(isinstance(value, str) for value in raw.reshape(-1).tolist())
    if has_text:
        labels = [str(value) for value in raw[:, 0].tolist()]
        encoded, class_labels = _encode_label_column(labels)
        return encoded, class_labels

    numeric = np.asarray(y, dtype=float)
    if numeric.ndim == 0:
        numeric = numeric.reshape(1, 1)
    elif numeric.ndim == 1:
        numeric = numeric.reshape(-1, 1)

    if numeric.shape[1] == 1:
        labels = []
        for value in numeric[:, 0].tolist():
            rounded = round(value)
            labels.append(int(rounded) if np.isclose(value, rounded, atol=1e-10) else value)
        encoded, class_labels = _encode_label_column(labels)
        return encoded, class_labels

    return numeric, None


def _print_predicted_classes(y_predicted, class_labels):
    predicted_indices = np.argmax(y_predicted, axis=1)
    print_value("y_predicted_classified", predicted_indices)
    if class_labels is not None:
        predicted_labels = [class_labels[index] for index in predicted_indices.tolist()]
        print_label_vector("y_predicted_class", predicted_labels)


def fit_onehot_linearclassification(X, y):
    y_encoded, class_labels = prepare_class_targets(y)
    system, w = calculate_closed_form_weights(X, y_encoded)
    print_system_info("X", X, system=system)
    print_weights(w)
    print("")

    y_calculated = X @ w
    sum_of_square, mean_squared_error = print_error_summary(y_encoded, y_calculated)
    return system, w, y_encoded, class_labels, sum_of_square, mean_squared_error


def predict_onehot_linearclassification(w, X_test, class_labels=None):
    y_predicted = X_test @ w
    print_value("y_predicted", y_predicted)
    _print_predicted_classes(y_predicted, class_labels)
    return y_predicted


def onehot_linearclassification(X, y, X_test=None):
    system, w, y_encoded, class_labels, sum_of_square, mean_squared_error = fit_onehot_linearclassification(X, y)
    y_predicted = None
    if X_test is not None:
        y_predicted = predict_onehot_linearclassification(w, X_test, class_labels=class_labels)
    return system, w, y_encoded, class_labels, sum_of_square, mean_squared_error, y_predicted


def fit_onehot_polynomialclassification(X, y, order):
    poly = PolynomialFeatures(order)
    P = poly.fit_transform(X)
    y_encoded, class_labels = prepare_class_targets(y)
    print("the number of parameters: ", P.shape[1])
    print("the number of samples: ", P.shape[0])
    system, w = calculate_closed_form_weights(P, y_encoded)
    print_system_info("P", P, system=system)
    print_weights(w)
    print("")

    y_calculated = P @ w
    sum_of_square, mean_squared_error = print_error_summary(y_encoded, y_calculated)
    return poly, system, w, y_encoded, class_labels, sum_of_square, mean_squared_error


def predict_onehot_polynomialclassification(poly, w, X_test, class_labels=None):
    P_test = poly.transform(X_test)
    print_value("P_test", P_test)
    y_predicted = P_test @ w
    print_value("y_predicted", y_predicted)
    _print_predicted_classes(y_predicted, class_labels)
    return y_predicted
