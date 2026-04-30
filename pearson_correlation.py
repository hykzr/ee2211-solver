import numpy as np

from CommonUtils import pretty_print_array


def _as_numeric_2d(value):
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def orient_pearson_inputs(X, Y):
    """Return X and Y with observations in rows and variables in columns."""
    X = _as_numeric_2d(X)
    Y = _as_numeric_2d(Y)

    if X.shape[0] == Y.shape[0]:
        if X.shape[0] == 1 and X.shape[1] == Y.shape[1] and X.shape[1] > 1:
            return X.T, Y.T, "single-row inputs transposed into observation rows"
        return X, Y, "observation rows"

    if X.shape[1] == Y.shape[1]:
        return X.T, Y.T, "matching columns transposed into observation rows"

    raise ValueError(
        "Pearson correlation expects matching observation counts: "
        f"X has {X.shape[0]} row(s) and {X.shape[1]} column(s), "
        f"Y has {Y.shape[0]} row(s) and {Y.shape[1]} column(s)."
    )


def pearson_correlation_details(X, Y):
    X, Y, orientation = orient_pearson_inputs(X, Y)
    if X.shape[0] < 2:
        raise ValueError("Pearson correlation needs at least two observations.")

    meanX = np.mean(X, axis=0, keepdims=True)
    standard_devX = np.std(X, axis=0, keepdims=True)
    var_X = np.var(X, axis=0, keepdims=True)
    meanY = np.mean(Y, axis=0, keepdims=True)
    standard_devY = np.std(Y, axis=0, keepdims=True)
    var_Y = np.var(Y, axis=0, keepdims=True)

    if np.any(np.isclose(standard_devX, 0)) or np.any(np.isclose(standard_devY, 0)):
        raise ValueError("Pearson correlation is undefined when X or Y has zero standard deviation.")

    centered_X = X - meanX
    centered_Y = Y - meanY
    covariance = centered_X.T @ centered_Y / X.shape[0]
    pearson = covariance / (standard_devX.T @ standard_devY)

    return {
        "X": X,
        "Y": Y,
        "orientation": orientation,
        "observations": X.shape[0],
        "x_variables": X.shape[1],
        "y_variables": Y.shape[1],
        "meanX": meanX,
        "stdX": standard_devX,
        "varX": var_X,
        "meanY": meanY,
        "stdY": standard_devY,
        "varY": var_Y,
        "covariance": covariance,
        "pearson": pearson,
    }


def pearson_correlation(X, Y):
    details = pearson_correlation_details(X, Y)
    pearson = details["pearson"]

    if details["orientation"] != "observation rows":
        print(f"Pearson input orientation: {details['orientation']}")
    print(f"observations: {details['observations']}")
    print(f"X variables: {details['x_variables']}")
    print(f"Y variables: {details['y_variables']}")
    print("")

    pretty_print_array("meanX", details["meanX"])
    pretty_print_array("stdX", details["stdX"])
    pretty_print_array("varX", details["varX"])
    pretty_print_array("meanY", details["meanY"])
    pretty_print_array("stdY", details["stdY"])
    pretty_print_array("varY", details["varY"])
    pretty_print_array("covariance", details["covariance"])
    pretty_print_array("pearson", pearson)
    return pearson
