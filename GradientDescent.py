import numpy as np
import sympy as sp

from CommonUtils import format_number, pretty_print_array


def infer_variables(cost_expression):
    expression = sp.sympify(cost_expression.replace("^", "**"))
    return [str(symbol) for symbol in sorted(expression.free_symbols, key=lambda item: item.name)]


def gradient_descent(cost_expression, variables=None, initial_values=None, learning_rate=0.1, iterations=10, tolerance=None):
    expression = sp.sympify(cost_expression.replace("^", "**"))
    if variables is None:
        variables = infer_variables(cost_expression)
    if not variables:
        raise ValueError("At least one variable is required.")

    symbols = [sp.Symbol(name.strip()) for name in variables]
    values = np.asarray(initial_values, dtype=float).reshape(-1)
    if values.size != len(symbols):
        raise ValueError(f"Expected {len(symbols)} initial values, got {values.size}.")

    gradient_expressions = [sp.diff(expression, symbol) for symbol in symbols]
    initial_gradient = _evaluate_gradient(gradient_expressions, symbols, values)
    history = [
        {
            "iteration": 0,
            "values": values.copy(),
            "gradient": initial_gradient.copy(),
            "cost": _evaluate_expression(expression, symbols, values),
        }
    ]

    for iteration in range(1, int(iterations) + 1):
        gradient = history[-1]["gradient"]
        next_values = values - float(learning_rate) * gradient
        next_cost = _evaluate_expression(expression, symbols, next_values)
        next_gradient = _evaluate_gradient(gradient_expressions, symbols, next_values)
        previous_cost = history[-1]["cost"]

        history.append(
            {
                "iteration": iteration,
                "values": next_values.copy(),
                "gradient": next_gradient.copy(),
                "cost": next_cost,
            }
        )

        parameter_change = np.max(np.abs(next_values - values))
        cost_change = abs(next_cost - previous_cost)
        values = next_values

        if tolerance is not None and (parameter_change <= tolerance or cost_change <= tolerance):
            break

    return expression, gradient_expressions, history


def print_gradient_descent_result(expression, variables, gradient_expressions, history):
    print(f"cost function: {sp.sstr(expression)}")
    for variable, gradient in zip(variables, gradient_expressions):
        print(f"dC/d{variable} = {sp.sstr(gradient)}")
    print("")

    header = ["iter", *variables, *(f"dC/d{variable}" for variable in variables), "cost"]
    print(" | ".join(header))
    print("-" * (len(" | ".join(header)) + 8))
    for item in history:
        row = [str(item["iteration"])]
        row.extend(format_number(value) for value in item["values"])
        row.extend(format_number(value) for value in item["gradient"])
        row.append(format_number(item["cost"]))
        print(" | ".join(row))

    final_values = history[-1]["values"].reshape(-1, 1)
    pretty_print_array("final_parameters", final_values)


def _evaluate_expression(expression, symbols, values):
    substitutions = {symbol: value for symbol, value in zip(symbols, values)}
    result=expression.evalf(subs=substitutions)
    try:
        return float(result)    
    except (TypeError, ValueError):
        raise ValueError(f"Failed to convert expression {result} to float")


def _evaluate_gradient(gradient_expressions, symbols, values):
    return np.array(
        [_evaluate_expression(grad_expr, symbols, values) for grad_expr in gradient_expressions],
        dtype=float,
    )
