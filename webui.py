import contextlib
import html
import io
import json
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import sympy as sp
from sklearn.preprocessing import PolynomialFeatures

from CommonUtils import (
    add_bias_column,
    calculate_closed_form_weights,
    calculate_ridge_weights,
    format_number,
    python_array_literal,
    squared_error_summary,
    validate_same_feature_count,
    validate_same_sample_count,
)
from DecisionTreeUtils import (
    CLASSIFICATION_MEASURES,
    classification_impurity_summary,
    find_best_regression_split,
    regression_threshold_summary,
)
from GradientDescent import gradient_descent, infer_variables
from OneHotLinearClassification import prepare_class_targets
from main import parse_mixed_array, parse_numeric_array
from pearson_correlation import pearson_correlation_details


MATRIX_HELP = (
    "Rows can be separated with semicolons or new lines. Examples: "
    "`1 2; 3 4`, `1, 2\\n3, 4`, or `[[1, 2], [3, 4]]`."
)
RIDGE_FORMS = ("auto", "primal form", "dual form")
STATE_FILE = Path(__file__).resolve().parent / "temp" / "webui_state.json"
STATE_VERSION = 1
PERSISTED_WIDGET_KEYS = (
    "array_inspect_raw",
    "solve_equation",
    "solve_x_raw",
    "solve_y_raw",
    "reg_model_type",
    "reg_target_mode",
    "reg_regularization",
    "reg_predict_test",
    "reg_order",
    "reg_include_bias",
    "reg_ridge_lambda",
    "reg_ridge_form",
    "reg_x_raw",
    "reg_y_raw",
    "reg_x_test_raw",
    "reg_y_test_raw",
    "pearson_x_raw",
    "pearson_y_raw",
    "gradient_cost_expression",
    "gradient_initial_values",
    "gradient_learning_rate",
    "gradient_iterations",
    "gradient_tolerance",
    "tree_mode",
    "tree_counts_raw",
    "tree_measure",
    "tree_x_raw",
    "tree_y_raw",
    "tree_threshold",
    "tree_find_best",
)


def main():
    configure_page()
    ensure_state()

    st.title("EE2211 Solver")

    tabs = st.tabs(
        [
            "Array",
            "Solve",
            "Regression",
            "Pearson",
            "Gradient",
            "Decision Tree",
            "Cache",
        ]
    )
    with tabs[0]:
        render_array_tab()
    with tabs[1]:
        render_solve_tab()
    with tabs[2]:
        render_regression_tab()
    with tabs[3]:
        render_pearson_tab()
    with tabs[4]:
        render_gradient_tab()
    with tabs[5]:
        render_decision_tree_tab()
    with tabs[6]:
        render_cache_tab()


def configure_page():
    st.set_page_config(page_title="EE2211 Solver", layout="wide", initial_sidebar_state="collapsed")
    st.markdown(
        """
        <style>
        :root {
          --solver-border: rgba(49, 51, 63, 0.16);
          --solver-soft: rgba(49, 51, 63, 0.055);
        }
        .block-container {
          padding-top: 2rem;
          padding-bottom: 3rem;
          max-width: 1320px;
        }
        h1 {
          font-size: 2rem;
          letter-spacing: 0;
          margin-bottom: 1rem;
        }
        h2, h3 {
          letter-spacing: 0;
        }
        div[data-testid="stMetric"] {
          border: 1px solid var(--solver-border);
          border-radius: 8px;
          padding: 0.45rem 0.65rem;
          background: var(--solver-soft);
          min-height: 0;
        }
        div[data-testid="stMetricLabel"] {
          font-size: 0.78rem;
          line-height: 1.15;
          overflow-wrap: anywhere;
        }
        div[data-testid="stMetricValue"] {
          font-size: 1.05rem !important;
          line-height: 1.2 !important;
          overflow-wrap: anywhere;
          white-space: normal;
        }
        div[data-testid="stMetricValue"] > div {
          font-size: inherit !important;
          line-height: inherit !important;
        }
        div[data-testid="stForm"] {
          border-radius: 8px;
          border-color: var(--solver-border);
        }
        .solver-meta {
          color: rgba(49, 51, 63, 0.74);
          font-size: 0.88rem;
          margin: -0.35rem 0 0.5rem 0;
        }
        .solver-chip {
          display: inline-block;
          border: 1px solid var(--solver-border);
          border-radius: 999px;
          padding: 0.1rem 0.45rem;
          margin-right: 0.25rem;
          font-size: 0.78rem;
          color: rgba(49, 51, 63, 0.76);
        }
        div[data-testid="stLatex"] {
          overflow-x: auto;
          margin: 0.1rem 0 0.45rem 0;
        }
        div[data-testid="stLatex"] .katex-display {
          margin: 0.15rem 0;
        }
        div[data-testid="stLatex"] .katex,
        div[data-testid="stLatex"] .katex-display > .katex {
          font-size: 1.05rem !important;
          line-height: 1.2;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def ensure_state():
    if "_solver_state_loaded" not in st.session_state:
        load_persisted_state()
        st.session_state._solver_state_loaded = True
    if "matrix_cache" not in st.session_state:
        st.session_state.matrix_cache = []


def load_persisted_state():
    try:
        payload = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(payload, dict):
        return

    widgets = payload.get("widgets", {})
    if isinstance(widgets, dict):
        for key in PERSISTED_WIDGET_KEYS:
            if key in widgets:
                st.session_state[key] = widgets[key]

    cache = payload.get("matrix_cache", [])
    if isinstance(cache, list):
        st.session_state.matrix_cache = [entry for entry in (deserialize_cache_entry(item) for item in cache) if entry]


def save_persisted_state():
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": STATE_VERSION,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "widgets": persisted_widget_values(),
            "matrix_cache": [
                serialize_cache_entry(entry)
                for entry in st.session_state.get("matrix_cache", [])
            ],
        }
        temp_file = STATE_FILE.with_suffix(".tmp")
        temp_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        temp_file.replace(STATE_FILE)
    except OSError:
        pass


def persisted_widget_values():
    return {key: json_safe_value(st.session_state[key]) for key in PERSISTED_WIDGET_KEYS if key in st.session_state}


def serialize_cache_entry(entry):
    value = as_2d_any(entry.get("value", []))
    return {
        "id": str(entry.get("id", uuid.uuid4().hex)),
        "name": str(entry.get("name", "")),
        "source": str(entry.get("source", "")),
        "kind": str(entry.get("kind", "")),
        "shape": str(entry.get("shape", shape_text(value))),
        "value": json_safe_value(value),
        "text": str(entry.get("text", matrix_plain_text(value))),
        "created_at": str(entry.get("created_at", "")),
    }


def deserialize_cache_entry(entry):
    if not isinstance(entry, dict):
        return None
    value = as_2d_any(entry.get("value", []))
    return {
        "id": str(entry.get("id") or uuid.uuid4().hex),
        "name": str(entry.get("name", "")),
        "source": str(entry.get("source", "")),
        "kind": str(entry.get("kind", "")),
        "shape": str(entry.get("shape") or shape_text(value)),
        "value": value,
        "text": str(entry.get("text") or matrix_plain_text(value)),
        "created_at": str(entry.get("created_at", "")),
    }


def json_safe_value(value):
    if isinstance(value, np.ndarray):
        return [[json_safe_value(cell) for cell in row] for row in as_2d_any(value).tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [json_safe_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_safe_value(item) for key, item in value.items()}
    return value


def render_array_tab():
    st.subheader("Parse And Inspect Array")
    with st.form("array_form", border=True):
        array, _ = matrix_input("Array", "array_inspect", default="1 2; 3 4")
        submitted = st.form_submit_button("Inspect")

    if submitted:
        save_persisted_state()
    if not submitted:
        return
    if array is None:
        st.error("Enter a valid numeric matrix first.")
        return

    record_matrix("array", "Array tab input", array, kind="input")
    st.markdown("#### Parsed Matrix")
    render_matrix("array", array)

    details, matrices = inspect_matrix(array)
    render_metrics(details)
    for name, value in matrices:
        st.markdown(f"#### {title_from_name(name)}")
        render_matrix(name, value)
        record_matrix(name, "Array inspection result", value, kind="output")


def render_solve_tab():
    st.subheader("Solve Linear System")
    with st.form("solve_form", border=True):
        equation = st.selectbox("Equation", ("Xw = y", "wX = y"), key="solve_equation")
        col_x, col_y = st.columns(2)
        with col_x:
            X, _ = matrix_input("X", "solve_x", default="1 1\n1 2\n1 3")
        with col_y:
            y, _ = matrix_input("y", "solve_y", default="2\n3\n4")
        submitted = st.form_submit_button("Solve")

    if submitted:
        save_persisted_state()
    if not submitted:
        return
    if X is None or y is None:
        st.error("Both X and y are required.")
        return

    try:
        result = solve_linear_system(X, y, equation)
    except Exception as exc:
        st.error(str(exc))
        return

    record_matrix("X", f"Solve {equation} input", X, kind="input")
    record_matrix("y", f"Solve {equation} input", y, kind="input")
    record_matrix("w", f"Solve {equation} result", result["w"], kind="trained matrix w")
    record_matrix("y_hat", f"Solve {equation} fitted target", result["predicted"], kind="output")
    record_matrix("residual", f"Solve {equation} residual", result["residual"], kind="output")
    record_matrix("mse", f"Solve {equation} error", result["mse"], kind="output")

    st.markdown("#### System")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Type", result["system"])
    c2.metric("Equations", result["equations"])
    c3.metric("Unknowns", result["unknowns"])
    c4.metric("Solution", result["status"])
    c1, c2 = st.columns(2)
    c1.metric("rank(coefficients)", result["rank_coefficients"])
    c2.metric("rank(augmented)", result["rank_augmented"])

    st.markdown("#### Solution")
    render_matrix("w", result["w"])
    st.markdown("#### Fitted Target")
    render_matrix(r"\hat{y}", result["predicted"])
    st.markdown("#### Error")
    render_matrix("residual", result["residual"])
    left, right = st.columns(2)
    with left:
        render_matrix("standard error", result["standard_error"])
    with right:
        render_matrix("mse", result["mse"])


def render_regression_tab():
    st.subheader("Regression And Classification")
    settings = st.columns(4)
    with settings[0]:
        model_type = st.selectbox("Model", ("Linear", "Polynomial"), key="reg_model_type")
    with settings[1]:
        target_mode = st.selectbox(
            "Y interpretation",
            ("Auto", "Numeric regression", "Sign classification", "One-hot classification"),
            key="reg_target_mode",
        )
    with settings[2]:
        regularization = st.selectbox("Regularization", ("None", "Ridge"), key="reg_regularization")
    with settings[3]:
        predict_test = st.checkbox("Predict X_test", value=False, key="reg_predict_test")

    order = 1
    include_bias = False
    ridge_lambda = 1.0
    ridge_form = RIDGE_FORMS[0]
    controls = st.columns(3)
    with controls[0]:
        if model_type == "Polynomial":
            order = st.number_input("Polynomial order", min_value=1, max_value=12, value=2, step=1, key="reg_order")
        else:
            include_bias = st.checkbox("Add bias for linear model", value=True, key="reg_include_bias")
    if regularization == "Ridge":
        with controls[1]:
            ridge_lambda = st.number_input("Lambda", min_value=0.0, value=1.0, step=0.1, key="reg_ridge_lambda")
        with controls[2]:
            ridge_form = st.selectbox("Ridge form", RIDGE_FORMS, key="reg_ridge_form")

    with st.form("regression_form", border=True):
        col_x, col_y = st.columns(2)
        with col_x:
            X, _ = matrix_input("X", "reg_x", default="0\n1\n2\n3")
        with col_y:
            allow_mixed_y = target_mode in {"Auto", "One-hot classification"}
            y_default = "1\n3\n5\n7" if target_mode == "Numeric regression" else "low\nlow\nhigh\nhigh"
            Y, _ = matrix_input("Y", "reg_y", default=y_default, allow_mixed=allow_mixed_y)

        X_test = None
        Y_test = None
        if predict_test:
            test_col_x, test_col_y = st.columns(2)
            with test_col_x:
                X_test, _ = matrix_input("X_test", "reg_x_test", default="4\n5")
            with test_col_y:
                Y_test, _ = matrix_input("Y_test (optional)", "reg_y_test", default="", allow_mixed=allow_mixed_y)

        submitted = st.form_submit_button("Fit")

    if submitted:
        save_persisted_state()
    if not submitted:
        return
    if X is None or Y is None:
        st.error("X and Y are required.")
        return
    if predict_test and X_test is None:
        st.error("X_test is enabled but no valid test matrix was provided.")
        return

    try:
        result = fit_regression_model(
            X=X,
            Y=Y,
            X_test=X_test,
            Y_test=Y_test,
            model_type=model_type,
            order=int(order),
            target_mode=target_mode,
            regularization=regularization,
            ridge_lambda=float(ridge_lambda),
            ridge_form=ridge_form,
            include_bias=bool(include_bias),
        )
    except Exception as exc:
        st.error(str(exc))
        return

    record_matrix("X", "Regression input", X, kind="input")
    record_matrix("Y", "Regression input", Y, kind="input")
    record_matrix(result["model_matrix_name"], "Regression model matrix", result["model_matrix"], kind="output")
    record_matrix("w", "Regression trained matrix w", result["w"], kind="trained matrix w")
    record_matrix("y_train_predicted", "Regression training prediction", result["train_pred"], kind="output")
    record_matrix("mse", "Regression training error", result["mse"], kind="output")
    if X_test is not None:
        record_matrix("X_test", "Regression prediction input", X_test, kind="input")
        record_matrix("y_predicted", "Regression test prediction", result["test_pred"], kind="output")
    if result["test_target_matrix"] is not None:
        record_matrix("Y_test", "Regression test target", Y_test, kind="input")
        record_matrix("test_mse", "Regression test error", result["test_mse"], kind="output")

    st.markdown("#### Model")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Model", result["model_label"])
    m2.metric("Target", result["target_mode"])
    m3.metric("System", result["system"])
    m4.metric("Form", result["form"])

    if result["class_labels"] is not None:
        st.markdown("#### Class Labels")
        st.write(", ".join(str(item) for item in result["class_labels"]))

    st.markdown("#### Model Matrix")
    render_matrix(result["model_matrix_name"], result["model_matrix"])
    if result["target_mode"] == "One-hot classification":
        st.markdown("#### Encoded Target")
        render_matrix("Y_encoded", result["target_matrix"])

    st.markdown("#### Weights")
    render_matrix("w", result["w"])
    st.markdown("#### Training Prediction")
    render_matrix(r"\hat{Y}_{train}", result["train_pred"])
    if result["train_class"] is not None:
        render_matrix("train_class", result["train_class"])

    left, right = st.columns(2)
    with left:
        render_matrix("square error", result["sse"])
    with right:
        render_matrix("mse", result["mse"])

    if X_test is not None:
        st.markdown("#### Test Prediction")
        render_matrix(r"\hat{Y}_{test}", result["test_pred"])
        if result["test_class"] is not None:
            render_matrix("test_class", result["test_class"])
        if result["test_target_matrix"] is not None:
            st.markdown("#### Test Error")
            if result["target_mode"] == "One-hot classification":
                render_matrix("Y_test_encoded", result["test_target_matrix"])
            left, right = st.columns(2)
            with left:
                render_matrix("test square error", result["test_sse"])
            with right:
                render_matrix("test mse", result["test_mse"])


def render_pearson_tab():
    st.subheader("Pearson Correlation")
    with st.form("pearson_form", border=True):
        col_x, col_y = st.columns(2)
        with col_x:
            X, _ = matrix_input("X", "pearson_x", default="1 1\n2 1\n3 2\n4 2")
        with col_y:
            Y, _ = matrix_input("Y", "pearson_y", default="2 1\n4 1\n6 2\n8 2")
        submitted = st.form_submit_button("Calculate")

    if submitted:
        save_persisted_state()
    if not submitted:
        return
    if X is None or Y is None:
        st.error("Both X and Y are required.")
        return

    try:
        details = pearson_details(X, Y)
    except Exception as exc:
        st.error(str(exc))
        return

    record_matrix("X", "Pearson input", X, kind="input")
    record_matrix("Y", "Pearson input", Y, kind="input")
    record_matrix("pearson", "Pearson result", details["pearson"], kind="output")

    st.markdown("#### Result")
    render_metrics(
        {
            "Observations": details["observations"],
            "X columns": details["x_variables"],
            "Y columns": details["y_variables"],
        }
    )
    render_matrix("pearson", details["pearson"])
    st.dataframe(pearson_frame(details["pearson"]), width="stretch")
    with st.expander("Intermediate values", expanded=True):
        for name in ("meanX", "stdX", "varX", "meanY", "stdY", "varY", "covariance"):
            render_matrix(name, details[name])


def render_gradient_tab():
    st.subheader("Gradient Descent")
    with st.form("gradient_form", border=True):
        cost_expression = st.text_input(
            "Cost function C(...)",
            value="x**2 + x*y**2",
            key="gradient_cost_expression",
        )
        inferred = infer_expression_variables(cost_expression)
        st.markdown(f'<div class="solver-meta">Variable order: {", ".join(inferred) or "none"}</div>', unsafe_allow_html=True)
        values_text = st.text_input("Initial values", value="3, 2", key="gradient_initial_values")
        controls = st.columns(3)
        with controls[0]:
            learning_rate = st.number_input("Learning rate", min_value=0.0, value=0.2, step=0.05, key="gradient_learning_rate")
        with controls[1]:
            iterations = st.number_input("Iterations", min_value=1, max_value=10000, value=10, step=1, key="gradient_iterations")
        with controls[2]:
            tolerance_text = st.text_input("Tolerance", value="", key="gradient_tolerance")
        submitted = st.form_submit_button("Run")

    if submitted:
        save_persisted_state()
    if not submitted:
        return
    if not cost_expression.strip():
        st.error("Cost function is required.")
        return

    try:
        variables = infer_variables(cost_expression)
        initial_values = parse_initial_values(values_text, expected=len(variables))
        tolerance = float(tolerance_text) if tolerance_text.strip() else None
        expression, gradient_expressions, history = gradient_descent(
            cost_expression,
            variables=variables,
            initial_values=initial_values,
            learning_rate=float(learning_rate),
            iterations=int(iterations),
            tolerance=tolerance,
        )
    except Exception as exc:
        st.error(str(exc))
        return

    final_values = history[-1]["values"].reshape(-1, 1)
    record_matrix("final_parameters", "Gradient descent result", final_values, kind="output")

    st.markdown("#### Function")
    st.latex(r"C = " + sp.latex(expression))
    for variable, gradient_expression in zip(variables, gradient_expressions):
        st.latex(rf"\frac{{\partial C}}{{\partial {latex_identifier(variable)}}} = {sp.latex(gradient_expression)}")

    st.markdown("#### History")
    st.dataframe(gradient_history_frame(history, variables), width="stretch", hide_index=True)
    st.markdown("#### Final Parameters")
    render_matrix("final_parameters", final_values)


def render_decision_tree_tab():
    st.subheader("Decision Tree")
    mode = st.selectbox("Mode", ("Classification impurity", "Regression MSE / split"), key="tree_mode")
    if mode == "Classification impurity":
        render_classification_tree_panel()
    else:
        render_regression_tree_panel()


def render_classification_tree_panel():
    with st.form("classification_tree_form", border=True):
        class_counts, _ = matrix_input("Class counts by node", "tree_counts", default="2 5\n6 0")
        measure = st.selectbox("Impurity measure", ("all", *CLASSIFICATION_MEASURES), key="tree_measure")
        submitted = st.form_submit_button("Calculate")

    if submitted:
        save_persisted_state()
    if not submitted:
        return
    if class_counts is None:
        st.error("Class counts are required.")
        return

    try:
        measures = CLASSIFICATION_MEASURES if measure == "all" else (measure,)
        summaries = []
        for item in measures:
            node_sizes, impurities, weighted_impurity = classification_impurity_summary(class_counts, item)
            summaries.append((item, node_sizes, impurities, weighted_impurity))
    except Exception as exc:
        st.error(str(exc))
        return

    record_matrix("class_counts", "Decision tree classification input", class_counts, kind="input")
    for item, node_sizes, impurities, weighted_impurity in summaries:
        record_matrix(f"{item}_per_node", "Decision tree impurity result", impurities.reshape(-1, 1), kind="output")
        record_matrix(f"weighted_{item}", "Decision tree impurity result", [[weighted_impurity]], kind="output")

    st.markdown("#### Class Counts")
    render_matrix("class_counts", class_counts)
    for item, node_sizes, impurities, weighted_impurity in summaries:
        st.markdown(f"#### {item.title()}")
        col1, col2, col3 = st.columns(3)
        with col1:
            render_matrix(f"{item}_per_node", impurities.reshape(-1, 1))
        with col2:
            render_matrix("node_sizes", node_sizes.reshape(-1, 1))
        with col3:
            render_matrix(f"weighted_{item}", [[weighted_impurity]])


def render_regression_tree_panel():
    with st.form("regression_tree_form", border=True):
        col_x, col_y = st.columns(2)
        with col_x:
            X, _ = matrix_input("x values", "tree_x", default="0.2\n0.7\n1.8\n2.2\n3.7\n4.1")
        with col_y:
            Y, _ = matrix_input("y values", "tree_y", default="2.1\n1.5\n5.8\n6.1\n9.1\n9.5")
        controls = st.columns(2)
        with controls[0]:
            threshold_text = st.text_input("Decision threshold", value="", key="tree_threshold")
        with controls[1]:
            find_best = st.checkbox("Find best one-level split", value=True, key="tree_find_best")
        submitted = st.form_submit_button("Calculate")

    if submitted:
        save_persisted_state()
    if not submitted:
        return
    if X is None or Y is None:
        st.error("Both x and y are required.")
        return

    threshold = None
    if threshold_text.strip():
        try:
            threshold = float(threshold_text)
        except ValueError:
            st.error("Decision threshold must be numeric.")
            return

    record_matrix("x", "Decision tree regression input", X, kind="input")
    record_matrix("y", "Decision tree regression input", Y, kind="input")

    if threshold is not None:
        try:
            summary = regression_threshold_summary(X, Y, threshold)
        except Exception as exc:
            st.error(str(exc))
            return
        record_matrix("threshold_summary", "Decision tree threshold result", threshold_summary_matrix(summary), kind="output")
        st.markdown("#### Threshold Summary")
        render_threshold_summary(summary)

    if find_best:
        try:
            best, summaries = find_best_regression_split(X, Y)
        except Exception as exc:
            st.error(str(exc))
            return
        record_matrix("best_split", "Decision tree best split result", [[best["threshold"]], [best["overall_mse"]]], kind="output")
        st.markdown("#### Candidate Splits")
        st.dataframe(split_candidates_frame(summaries), width="stretch", hide_index=True)
        st.markdown("#### Best Split")
        render_threshold_summary(best)


def render_cache_tab():
    st.subheader("Cache")
    cache = st.session_state.matrix_cache
    top_left, top_right = st.columns([1, 3])
    with top_left:
        if st.button("Clear cache", disabled=not cache, width="stretch"):
            st.session_state.matrix_cache = []
            save_persisted_state()
            st.rerun()
    with top_right:
        st.markdown(
            f'<div class="solver-meta">{len(cache)} cached matrix item{"s" if len(cache) != 1 else ""}</div>',
            unsafe_allow_html=True,
        )

    if not cache:
        st.info("No matrices have been cached yet.")
        return

    for entry in cache:
        with st.container(border=True):
            header_cols = st.columns([3, 2, 1])
            with header_cols[0]:
                st.markdown(f"#### {entry['name']}")
                st.markdown(
                    f'<span class="solver-chip">{entry["kind"]}</span>'
                    f'<span class="solver-chip">{entry["shape"]}</span>',
                    unsafe_allow_html=True,
                )
            with header_cols[1]:
                st.markdown(f'<div class="solver-meta">{html.escape(entry["source"])}</div>', unsafe_allow_html=True)
            with header_cols[2]:
                if st.button("Remove", key=f"remove_{entry['id']}", width="stretch"):
                    remove_cache_item(entry["id"])
                    st.rerun()

            render_matrix(entry["name"], entry["value"])
            with st.expander("Plain text"):
                st.code(entry["text"], language="python")


def matrix_input(label, key, default="", allow_mixed=False):
    raw_key = f"{key}_raw"
    if raw_key not in st.session_state:
        st.session_state[raw_key] = default

    raw = st.text_area(label, key=raw_key, help=MATRIX_HELP, height=110)
    value = None
    status = "empty"
    if raw.strip():
        try:
            value = parse_matrix(raw, allow_mixed=allow_mixed)
            status = "typed"
        except Exception as exc:
            st.warning(f"{label}: {exc}")
            status = "invalid"

    st.form_submit_button(
        f"Transpose {label}",
        key=f"{key}_transpose",
        disabled=value is None,
        on_click=transpose_matrix_input,
        args=(raw_key, allow_mixed),
        width="stretch",
    )

    if value is None:
        return None, status

    with st.expander(f"Preview {label}", expanded=False):
        render_matrix(label, value, copy=False)
    return value, status


def transpose_matrix_input(raw_key, allow_mixed):
    try:
        value = parse_matrix(st.session_state.get(raw_key, ""), allow_mixed=allow_mixed)
    except Exception:
        return
    st.session_state[raw_key] = matrix_editor_text(as_2d_any(value).T)
    save_persisted_state()


def parse_matrix(raw_text, allow_mixed=False):
    text = normalize_matrix_text(raw_text)
    parser = parse_mixed_array if allow_mixed else parse_numeric_array
    return parser(text)


def normalize_matrix_text(raw_text):
    text = raw_text.strip().replace("−", "-").replace("–", "-").replace("Ã¢Ë†â€™", "-")
    if not text:
        return text
    if text.startswith(("[", "(")):
        return " ".join(line.strip() for line in text.splitlines() if line.strip())
    return "; ".join(line.strip() for line in text.splitlines() if line.strip())


def inspect_matrix(array):
    matrix = numeric_matrix(array)
    rows, cols = matrix.shape
    rank = int(np.linalg.matrix_rank(matrix))
    is_square = rows == cols
    inverse_exists = is_square and rank == rows
    left_inverse_exists = rank == cols
    right_inverse_exists = rank == rows

    details = {
        "shape": f"{rows} x {cols}",
        "rank": rank,
        "determinant": format_number(np.linalg.det(matrix)) if is_square else "n/a",
        "inverse": "yes" if inverse_exists else "no",
        "left inverse": "yes" if left_inverse_exists else "no",
        "right inverse": "yes" if right_inverse_exists else "no",
    }
    matrices = []
    if inverse_exists:
        matrices.append(("inverse", np.linalg.inv(matrix)))
    if left_inverse_exists:
        matrices.append(("left_inverse", np.linalg.inv(matrix.T @ matrix) @ matrix.T))
    if right_inverse_exists:
        matrices.append(("right_inverse", matrix.T @ np.linalg.inv(matrix @ matrix.T)))
    return details, matrices


def solve_linear_system(X, y, equation):
    X = numeric_matrix(X)
    y = numeric_matrix(y)
    if equation == "Xw = y":
        validate_same_sample_count(X, y)
        coefficients = X
        target = y
        orient = lambda value: value
    else:
        if X.shape[1] != y.shape[1]:
            raise ValueError(f"Column count mismatch: X has {X.shape[1]} columns but y has {y.shape[1]} columns.")
        coefficients = X.T
        target = y.T
        orient = lambda value: value.T

    rows, variables = coefficients.shape
    rank_coefficients = int(np.linalg.matrix_rank(coefficients))
    rank_augmented = int(np.linalg.matrix_rank(np.hstack((coefficients, target))))
    is_consistent = rank_coefficients == rank_augmented
    solution, _, _, _ = np.linalg.lstsq(coefficients, target, rcond=None)
    predicted = coefficients @ solution
    residual = predicted - target
    mse = np.mean(np.square(residual), axis=0, keepdims=True)
    standard_error = np.sqrt(mse)

    if is_consistent and rank_coefficients == variables:
        status = "unique exact"
    elif is_consistent:
        status = "many exact"
    else:
        status = "least squares"

    return {
        "system": system_shape_description(coefficients),
        "equations": rows,
        "unknowns": variables,
        "rank_coefficients": rank_coefficients,
        "rank_augmented": rank_augmented,
        "status": status,
        "w": orient(solution),
        "predicted": orient(predicted),
        "residual": orient(residual),
        "standard_error": orient(standard_error),
        "mse": orient(mse),
    }


def fit_regression_model(
    X,
    Y,
    X_test,
    Y_test,
    model_type,
    order,
    target_mode,
    regularization,
    ridge_lambda,
    ridge_form,
    include_bias,
):
    X = numeric_matrix(X)
    validate_same_sample_count(X, as_2d_any(Y))

    model_matrix, poly, model_matrix_name, model_label = build_model_matrix(X, model_type, order, include_bias)
    resolved_target_mode = resolve_target_mode(Y, target_mode)
    target_matrix, class_labels = prepare_target_matrix(Y, resolved_target_mode)

    if regularization == "Ridge":
        system, resolved_form, w = calculate_ridge_weights(
            model_matrix,
            target_matrix,
            ridge_lambda=ridge_lambda,
            form=ridge_form,
        )
        form = f"{resolved_form}, lambda={format_number(ridge_lambda)}"
    else:
        system, w = calculate_closed_form_weights(model_matrix, target_matrix)
        form = "closed form"

    train_pred = model_matrix @ w
    sse, mse = squared_error_summary(target_matrix, train_pred)
    train_class = classify_predictions(train_pred, resolved_target_mode, class_labels)

    test_pred = None
    test_class = None
    test_target_matrix = None
    test_sse = None
    test_mse = None
    if X_test is not None:
        X_test = numeric_matrix(X_test)
        validate_same_feature_count(X, X_test)
        X_test_model = transform_test_matrix(X_test, model_type, include_bias, poly)
        test_pred = X_test_model @ w
        test_class = classify_predictions(test_pred, resolved_target_mode, class_labels)
        if Y_test is not None:
            test_target_matrix = prepare_test_target_matrix(Y_test, resolved_target_mode, class_labels)
            validate_same_sample_count(X_test, test_target_matrix)
            if test_target_matrix.shape[1] != test_pred.shape[1]:
                raise ValueError(
                    f"Test target output count mismatch: predictions have {test_pred.shape[1]} column(s) "
                    f"but Y_test has {test_target_matrix.shape[1]} column(s)."
                )
            test_sse, test_mse = squared_error_summary(test_target_matrix, test_pred)

    return {
        "model_label": model_label,
        "model_matrix_name": model_matrix_name,
        "model_matrix": model_matrix,
        "target_mode": resolved_target_mode,
        "target_matrix": target_matrix,
        "class_labels": class_labels,
        "system": system,
        "form": form,
        "w": w,
        "train_pred": train_pred,
        "train_class": train_class,
        "sse": np.asarray(sse).reshape(1, -1),
        "mse": np.asarray(mse).reshape(1, -1),
        "test_pred": test_pred,
        "test_class": test_class,
        "test_target_matrix": test_target_matrix,
        "test_sse": None if test_sse is None else np.asarray(test_sse).reshape(1, -1),
        "test_mse": None if test_mse is None else np.asarray(test_mse).reshape(1, -1),
    }


def build_model_matrix(X, model_type, order, include_bias):
    if model_type == "Polynomial":
        poly = PolynomialFeatures(order)
        P = poly.fit_transform(X)
        return P, poly, "P", f"Polynomial order {order}"

    poly = None
    X_model = add_bias_column(X) if include_bias else X
    model_label = "Linear + bias" if include_bias else "Linear"
    return X_model, poly, "X_model", model_label


def transform_test_matrix(X_test, model_type, include_bias, poly):
    if model_type == "Polynomial":
        return poly.transform(X_test)
    return add_bias_column(X_test) if include_bias else X_test


def resolve_target_mode(Y, requested):
    if requested != "Auto":
        return requested

    arr = as_2d_any(Y)
    if contains_text(arr):
        return "One-hot classification"

    numeric = numeric_matrix(arr)
    unique = set(np.unique(numeric).tolist())
    if numeric.shape[1] == 1 and unique.issubset({-1.0, 1.0}):
        return "Sign classification"
    if numeric.shape[1] > 1:
        return "One-hot classification"
    return "Numeric regression"


def prepare_target_matrix(Y, target_mode):
    if target_mode == "One-hot classification":
        return prepare_class_targets(Y)
    return numeric_matrix(Y), None


def prepare_test_target_matrix(Y_test, target_mode, class_labels):
    if target_mode != "One-hot classification":
        return numeric_matrix(Y_test)

    arr = as_2d_any(Y_test)
    if arr.shape[1] > 1:
        return numeric_matrix(arr)
    if class_labels is None:
        encoded, _ = prepare_class_targets(Y_test)
        return encoded

    class_to_index = {label: index for index, label in enumerate(class_labels)}
    labels = arr.reshape(-1).tolist()
    encoded = np.zeros((len(labels), len(class_labels)), dtype=float)
    for row_index, label in enumerate(labels):
        if label not in class_to_index:
            raise ValueError(f"Y_test contains an unknown class label: {label!r}.")
        encoded[row_index, class_to_index[label]] = 1.0
    return encoded


def classify_predictions(prediction, target_mode, class_labels):
    if target_mode == "Sign classification":
        signs = np.sign(prediction)
        signs[np.isclose(signs, 0)] = 0
        return signs
    if target_mode == "One-hot classification":
        indices = np.argmax(prediction, axis=1).reshape(-1, 1)
        if class_labels is None:
            return indices
        labels = [class_labels[index] for index in indices.reshape(-1).tolist()]
        return np.asarray(labels, dtype=object).reshape(-1, 1)
    return None


def pearson_details(X, Y):
    return pearson_correlation_details(X, Y)


def infer_expression_variables(cost_expression):
    try:
        return infer_variables(cost_expression) if cost_expression.strip() else []
    except Exception:
        return []


def parse_initial_values(raw_text, expected):
    parts = [part.strip() for part in raw_text.replace("−", "-").replace("–", "-").split(",")]
    if any(not part for part in parts):
        raise ValueError("Enter one initial value per variable, separated by commas.")
    if len(parts) != expected:
        raise ValueError(f"Expected {expected} initial value(s), got {len(parts)}.")
    return np.asarray([float(part) for part in parts], dtype=float)


def gradient_history_frame(history, variables):
    rows = []
    for item in history:
        row = {"iteration": item["iteration"]}
        for variable, value in zip(variables, item["values"]):
            row[variable] = format_number(value)
        for variable, value in zip(variables, item["gradient"]):
            row[f"dC/d{variable}"] = "" if np.isnan(value) else format_number(value)
        row["cost"] = format_number(item["cost"])
        rows.append(row)
    return pd.DataFrame(rows)


def render_threshold_summary(summary):
    metrics = {
        "threshold": summary["threshold"],
        "root mean": summary["root_mean"],
        "root mse": summary["root_mse"],
        "left mean": summary["left_mean"],
        "left mse": summary["left_mse"],
        "right mean": summary["right_mean"],
        "right mse": summary["right_mse"],
        "split mse": summary["overall_mse"],
    }
    render_metrics({key: format_number(value) for key, value in metrics.items()})
    render_matrix("summary", threshold_summary_matrix(summary))


def threshold_summary_matrix(summary):
    return np.asarray(
        [
            [summary["threshold"]],
            [summary["root_mse"]],
            [summary["left_mse"]],
            [summary["right_mse"]],
            [summary["overall_mse"]],
        ],
        dtype=float,
    )


def split_candidates_frame(summaries):
    return pd.DataFrame(
        [
            {
                "threshold": format_number(summary["threshold"]),
                "overall_mse": format_number(summary["overall_mse"]),
                "left_mse": format_number(summary["left_mse"]),
                "right_mse": format_number(summary["right_mse"]),
            }
            for summary in summaries
        ]
    )


def render_matrix(name, matrix, compact=False, show_table=False, copy=True):
    arr = as_2d_any(matrix)
    if compact:
        st.markdown(f'<div class="solver-meta">{name}: {shape_text(arr)}</div>', unsafe_allow_html=True)
    if copy:
        matrix_cols = st.columns([11, 1.4])
        with matrix_cols[0]:
            st.latex(matrix_latex(name, arr))
        with matrix_cols[1]:
            clipboard_button(matrix_plain_text(arr), "Copy", uuid.uuid4().hex)
    else:
        st.latex(matrix_latex(name, arr))
    if show_table:
        st.dataframe(matrix_frame(arr), width="stretch")


def render_metrics(items):
    columns = st.columns(min(4, max(1, len(items))))
    for index, (label, value) in enumerate(items.items()):
        columns[index % len(columns)].metric(label, str(value))


def pearson_frame(matrix):
    arr = np.asarray(as_2d_any(matrix), dtype=float)
    return pd.DataFrame(
        [[format_number(value) for value in row] for row in arr.tolist()],
        index=[f"X c{i + 1}" for i in range(arr.shape[0])],
        columns=[f"Y c{i + 1}" for i in range(arr.shape[1])],
    )


def matrix_latex(name, matrix):
    arr = as_2d_any(matrix)
    rows = []
    for row in arr.tolist():
        rows.append(" & ".join(latex_cell(value) for value in row))
    body = r" \\ ".join(rows) if rows else ""
    return rf"{latex_identifier(name)} = \begin{{bmatrix}} {body} \end{{bmatrix}}"


def latex_cell(value):
    if is_numeric_scalar(value):
        return format_number(float(value))
    text = str(value)
    escaped = (
        text.replace("\\", r"\textbackslash{}")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("_", r"\_")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("#", r"\#")
        .replace("$", r"\$")
    )
    return rf"\text{{{escaped}}}"


def latex_identifier(name):
    replacements = {
        r"\hat{y}": r"\hat{y}",
        r"\hat{Y}_{train}": r"\hat{Y}_{train}",
        r"\hat{Y}_{test}": r"\hat{Y}_{test}",
    }
    if name in replacements:
        return replacements[name]
    text = str(name)
    escaped = (
        text.replace("\\", "")
        .replace("{", "")
        .replace("}", "")
        .replace("_", r"\_")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("#", r"\#")
        .replace("$", r"\$")
        .replace(" ", r"\ ")
    )
    if text.replace("_", "").isalnum():
        return escaped
    return rf"\mathrm{{{escaped}}}"


def matrix_frame(matrix):
    arr = as_2d_any(matrix)
    return pd.DataFrame(
        [[plain_cell(value) for value in row] for row in arr.tolist()],
        index=[f"r{i + 1}" for i in range(arr.shape[0])],
        columns=[f"c{i + 1}" for i in range(arr.shape[1])],
    )


def plain_cell(value):
    if is_numeric_scalar(value):
        return format_number(float(value))
    return str(value)


def title_from_name(name):
    return str(name).replace("_", " ").title()


def as_2d_any(value):
    arr = np.asarray(value, dtype=object)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def numeric_matrix(value):
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def contains_text(matrix):
    return any(isinstance(value, str) for value in as_2d_any(matrix).reshape(-1).tolist())


def is_numeric_scalar(value):
    try:
        float(value)
        return not isinstance(value, str)
    except (TypeError, ValueError):
        return False


def system_shape_description(matrix):
    rows, cols = matrix.shape
    if rows > cols:
        return "overdetermined"
    if rows < cols:
        return "underdetermined"
    return "evendetermined"


def record_matrix(name, source, value, kind):
    arr = as_2d_any(value)
    st.session_state.matrix_cache.insert(
        0,
        {
            "id": uuid.uuid4().hex,
            "name": str(name),
            "source": str(source),
            "kind": str(kind),
            "shape": shape_text(arr),
            "value": arr.copy(),
            "text": matrix_plain_text(arr),
            "created_at": time.strftime("%H:%M:%S"),
        },
    )
    save_persisted_state()


def remove_cache_item(entry_id):
    st.session_state.matrix_cache = [entry for entry in st.session_state.matrix_cache if entry["id"] != entry_id]
    save_persisted_state()


def cache_label(entry):
    return f"{entry['name']} | {entry['source']} | {entry['shape']}"


def shape_text(matrix):
    arr = as_2d_any(matrix)
    return f"{arr.shape[0]} x {arr.shape[1]}"


def matrix_editor_text(matrix):
    arr = as_2d_any(matrix)
    if all(is_numeric_scalar(value) for value in arr.reshape(-1).tolist()):
        return "\n".join(" ".join(format_number(float(value)) for value in row) for row in arr.tolist())
    return json.dumps(arr.tolist(), ensure_ascii=False)


def matrix_plain_text(matrix):
    arr = as_2d_any(matrix)
    if all(is_numeric_scalar(value) for value in arr.reshape(-1).tolist()):
        return python_array_literal(np.asarray(arr, dtype=float))
    return json.dumps(arr.tolist(), ensure_ascii=False)


def clipboard_button(text, label, entry_id):
    button_id = f"copy_{entry_id}"
    payload = json.dumps(text).replace("</", "<\\/")
    reset_label = json.dumps(label).replace("</", "<\\/")
    safe_label = html.escape(label)
    st.iframe(
        f"""
        <style>
        html, body {{
            margin: 0;
            padding: 0;
            width: 100%;
            height: 100%;
            overflow: hidden;
            background: transparent;
        }}
        #{button_id} {{
            width: 100%;
            height: 32px;
            box-sizing: border-box;
            border: 1px solid rgba(49, 51, 63, 0.18);
            border-radius: 6px;
            padding: 0 0.55rem;
            margin: 0;
            background: #ffffff;
            color: #111827;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font: 13px system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            line-height: 1;
            white-space: nowrap;
        }}
        #{button_id}:hover {{
            border-color: rgba(49, 51, 63, 0.32);
            background: #f8fafc;
        }}
        </style>
        <button id="{button_id}">{safe_label}</button>
        <script>
        const button = document.getElementById("{button_id}");
        button.addEventListener("click", async () => {{
            try {{
                await navigator.clipboard.writeText({payload});
                const original = button.textContent;
                button.textContent = "Copied";
                setTimeout(() => button.textContent = original, 900);
            }} catch (error) {{
                button.textContent = "Copy failed";
                setTimeout(() => button.textContent = {reset_label}, 1200);
            }}
        }});
        </script>
        """,
        height=32,
    )


@contextlib.contextmanager
def quiet_stdout():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


if __name__ == "__main__":
    main()
