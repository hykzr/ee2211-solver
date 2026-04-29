import numpy as np

from CommonUtils import ensure_2d, format_number, pretty_print_array, print_value


CLASSIFICATION_MEASURES = ("gini", "entropy", "misclassification")


def class_probabilities(class_counts):
    counts = np.asarray(class_counts, dtype=float).reshape(-1)
    if np.any(counts < 0):
        raise ValueError("Class counts cannot be negative.")
    total = np.sum(counts)
    if total <= 0:
        raise ValueError("A node must contain at least one sample.")
    return counts / total


def classification_impurity(class_counts, measure="gini"):
    probabilities = class_probabilities(class_counts)
    measure = measure.lower()
    if measure == "gini":
        return 1.0 - np.sum(np.square(probabilities))
    if measure == "entropy":
        nonzero = probabilities[probabilities > 0]
        return -np.sum(nonzero * np.log2(nonzero))
    if measure == "misclassification":
        return 1.0 - np.max(probabilities)
    raise ValueError(f"Unknown impurity measure: {measure}.")


def classification_impurity_summary(class_counts_by_node, measure="gini"):
    counts = ensure_2d(class_counts_by_node)
    node_sizes = np.sum(counts, axis=1)
    if np.any(node_sizes <= 0):
        raise ValueError("Every node row must contain at least one sample.")

    impurities = np.array([classification_impurity(row, measure=measure) for row in counts], dtype=float)
    weighted_impurity = float(np.sum(node_sizes / np.sum(node_sizes) * impurities))
    return node_sizes, impurities, weighted_impurity


def regression_node_mse(y):
    values = np.asarray(y, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("A regression-tree node must contain at least one sample.")
    mean = float(np.mean(values))
    mse = float(np.mean(np.square(values - mean)))
    return mean, mse


def regression_split_summary(y_left, y_right):
    left = np.asarray(y_left, dtype=float).reshape(-1)
    right = np.asarray(y_right, dtype=float).reshape(-1)
    left_mean, left_mse = regression_node_mse(left)
    right_mean, right_mse = regression_node_mse(right)
    total = left.size + right.size
    overall_mse = float((left.size * left_mse + right.size * right_mse) / total)
    return {
        "left_size": left.size,
        "right_size": right.size,
        "left_mean": left_mean,
        "right_mean": right_mean,
        "left_mse": left_mse,
        "right_mse": right_mse,
        "overall_mse": overall_mse,
    }


def regression_threshold_summary(x, y, threshold):
    x_values, y_values = _prepare_xy(x, y)
    left_mask = x_values <= threshold
    right_mask = x_values > threshold
    if not np.any(left_mask) or not np.any(right_mask):
        raise ValueError("Threshold must leave at least one sample on each side.")

    root_mean, root_mse = regression_node_mse(y_values)
    split = regression_split_summary(y_values[left_mask], y_values[right_mask])
    split.update(
        {
            "threshold": float(threshold),
            "root_mean": root_mean,
            "root_mse": root_mse,
            "x_sorted": x_values,
            "y_sorted": y_values,
        }
    )
    return split


def find_best_regression_split(x, y):
    x_values, y_values = _prepare_xy(x, y)
    unique_x = np.unique(x_values)
    if unique_x.size < 2:
        raise ValueError("At least two unique x values are needed to split.")

    thresholds = (unique_x[:-1] + unique_x[1:]) / 2.0
    summaries = [regression_threshold_summary(x_values, y_values, threshold) for threshold in thresholds]
    best = min(summaries, key=lambda item: item["overall_mse"])
    return best, summaries


def print_classification_impurity_summary(class_counts_by_node, measure="gini"):
    measures = CLASSIFICATION_MEASURES if measure == "all" else (measure,)
    pretty_print_array("class_counts_by_node", class_counts_by_node)
    for item in measures:
        node_sizes, impurities, weighted_impurity = classification_impurity_summary(class_counts_by_node, item)
        pretty_print_array(f"{item}_per_node", impurities.reshape(-1, 1), show_python=False)
        pretty_print_array("node_sizes", node_sizes.reshape(-1, 1), show_python=False)
        print_value(f"weighted_{item}", weighted_impurity)
        print("")


def print_regression_threshold_summary(summary):
    print_value("threshold", summary["threshold"])
    print_value("root_mean", summary["root_mean"])
    print_value("root_mse", summary["root_mse"])
    print_value("left_mean", summary["left_mean"])
    print_value("left_mse", summary["left_mse"])
    print_value("right_mean", summary["right_mean"])
    print_value("right_mse", summary["right_mse"])
    print_value("split_overall_mse", summary["overall_mse"])
    print("")


def print_regression_split_candidates(summaries):
    print("threshold | overall_mse")
    print("-----------------------")
    for summary in summaries:
        print(f"{format_number(summary['threshold'])} | {format_number(summary['overall_mse'])}")
    print("")


def _prepare_xy(x, y):
    x_values = np.asarray(x, dtype=float).reshape(-1)
    y_values = np.asarray(y, dtype=float).reshape(-1)
    if x_values.size != y_values.size:
        raise ValueError(f"x and y must have the same length, got {x_values.size} and {y_values.size}.")
    if x_values.size < 2:
        raise ValueError("At least two samples are needed.")
    sort_index = np.argsort(x_values, kind="mergesort")
    return x_values[sort_index], y_values[sort_index]
