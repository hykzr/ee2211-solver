import numpy as np

from CommonUtils import pretty_print_array


def pearson_correlation(X, Y):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if Y.ndim == 1:
        Y = Y.reshape(1, -1)
    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"Pearson correlation expects matching observation counts: X has {X.shape[1]}, Y has {Y.shape[1]}."
        )

    meanX = np.mean(X, axis=1, keepdims=True)
    standard_devX = np.std(X, axis=1, keepdims=True)
    var_X = np.var(X, axis=1, keepdims=True)
    meanY = np.mean(Y, axis=1, keepdims=True)
    standard_devY = np.std(Y, axis=1, keepdims=True)
    var_Y = np.var(Y, axis=1, keepdims=True)

    if np.any(np.isclose(standard_devX, 0)) or np.any(np.isclose(standard_devY, 0)):
        raise ValueError("Pearson correlation is undefined when X or Y has zero standard deviation.")

    centered_X = X - meanX
    centered_Y = Y - meanY
    covariance = centered_X @ centered_Y.T / X.shape[1]
    pearson = covariance / (standard_devX @ standard_devY.T)

    pretty_print_array("meanX", meanX)
    pretty_print_array("stdX", standard_devX)
    pretty_print_array("varX", var_X)
    pretty_print_array("meanY", meanY)
    pretty_print_array("stdY", standard_devY)
    pretty_print_array("varY", var_Y)
    pretty_print_array("covariance", covariance)
    pretty_print_array("pearson", pearson)
    return pearson
