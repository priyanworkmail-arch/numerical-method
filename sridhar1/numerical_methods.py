"""Numerical methods: Lagrange, Newton forward/backward, RK4, Modified Euler."""

from __future__ import annotations

import math
import re
from typing import Callable


def parse_number_list(text: str) -> list[float]:
    """Parse comma or space separated numbers."""
    parts = re.split(r"[,;\s]+", text.strip())
    return [float(p) for p in parts if p]


def parse_points(text: str) -> tuple[list[float], list[float]]:
    """Parse x,y pairs like '1,2; 3,4' or 'x: 1 2 3 y: 4 5 6'."""
    text = text.strip()
    x_match = re.search(r"x\s*[:=]\s*([\d\s,.\-eE+]+)", text, re.I)
    y_match = re.search(r"y\s*[:=]\s*([\d\s,.\-eE+]+)", text, re.I)
    if x_match and y_match:
        xs = parse_number_list(x_match.group(1))
        ys = parse_number_list(y_match.group(1))
        if len(xs) != len(ys):
            raise ValueError("x and y must have the same number of values.")
        return xs, ys

    pairs = re.findall(r"([\d.\-eE+]+)\s*,\s*([\d.\-eE+]+)", text)
    if pairs:
        xs = [float(a) for a, _ in pairs]
        ys = [float(b) for _, b in pairs]
        return xs, ys

    nums = parse_number_list(text)
    if len(nums) % 2 != 0:
        raise ValueError("Provide pairs as 'x1,y1; x2,y2' or separate x and y lists.")
    xs = nums[0::2]
    ys = nums[1::2]
    return xs, ys


def lagrange_interpolation(xs: list[float], ys: list[float], x: float) -> float:
    n = len(xs)
    result = 0.0
    for i in range(n):
        term = ys[i]
        for j in range(n):
            if i != j:
                term *= (x - xs[j]) / (xs[i] - xs[j])
        result += term
    return result


def _forward_differences(ys: list[float]) -> list[list[float]]:
    table = [ys[:]]
    n = len(ys)
    for k in range(1, n):
        prev = table[-1]
        table.append([prev[i + 1] - prev[i] for i in range(n - k)])
    return table


def _backward_differences(ys: list[float]) -> list[list[float]]:
    table = [ys[:]]
    n = len(ys)
    for k in range(1, n):
        prev = table[-1]
        table.append([prev[i + 1] - prev[i] for i in range(n - k)])
    return table


def newton_forward_interpolation(
    xs: list[float], ys: list[float], x: float
) -> tuple[float, str]:
    n = len(xs)
    h = xs[1] - xs[0]
    for i in range(1, n):
        if not math.isclose(xs[i] - xs[i - 1], h, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("Forward interpolation requires equally spaced x values.")

    u = (x - xs[0]) / h
    diffs = _forward_differences(ys)
    result = ys[0]
    term = 1.0
    fact = 1
    details = [f"f(x0) = {ys[0]}"]
    for r in range(1, n):
        fact *= r
        term *= u - (r - 1)
        delta = diffs[r][0]
        contribution = (term / fact) * delta
        result += contribution
        details.append(f"+ (Δ^{r}y0 / {fact}) * u(u-1)...(u-{r-1}) = {contribution:.8g}")
    return result, "\n".join(details)


def newton_backward_interpolation(
    xs: list[float], ys: list[float], x: float
) -> tuple[float, str]:
    n = len(xs)
    h = xs[1] - xs[0]
    for i in range(1, n):
        if not math.isclose(xs[i] - xs[i - 1], h, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("Backward interpolation requires equally spaced x values.")

    u = (x - xs[-1]) / h
    diffs = _backward_differences(ys)
    result = ys[-1]
    term = 1.0
    fact = 1
    details = [f"f(xn) = {ys[-1]}"]
    for r in range(1, n):
        fact *= r
        term *= u + (r - 1)
        delta = diffs[r][-1]
        contribution = (term / fact) * delta
        result += contribution
        details.append(f"+ (∇^{r}yn / {fact}) * u(u+1)...(u+{r-1}) = {contribution:.8g}")
    return result, "\n".join(details)


def runge_kutta_4(
    f: Callable[[float, float], float],
    x0: float,
    y0: float,
    h: float,
    x_end: float,
) -> tuple[list[tuple[float, float]], str]:
    steps: list[tuple[float, float]] = [(x0, y0)]
    x, y = x0, y0
    lines = [f"{'x':>12} {'y':>14} {'k1':>12} {'k2':>12} {'k3':>12} {'k4':>12}"]
    while x < x_end - 1e-12:
        if x + h > x_end:
            h = x_end - x
        k1 = h * f(x, y)
        k2 = h * f(x + h / 2, y + k1 / 2)
        k3 = h * f(x + h / 2, y + k2 / 2)
        k4 = h * f(x + h, y + k3)
        y = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        x = x + h
        steps.append((x, y))
        lines.append(
            f"{x:12.6f} {y:14.8f} {k1:12.6f} {k2:12.6f} {k3:12.6f} {k4:12.6f}"
        )
    return steps, "\n".join(lines)


def modified_euler(
    f: Callable[[float, float], float],
    x0: float,
    y0: float,
    h: float,
    x_end: float,
) -> tuple[list[tuple[float, float]], str]:
    steps: list[tuple[float, float]] = [(x0, y0)]
    x, y = x0, y0
    lines = [f"{'x':>12} {'y_pred':>14} {'y_corr':>14}"]
    while x < x_end - 1e-12:
        if x + h > x_end:
            h = x_end - x
        y_pred = y + h * f(x, y)
        y_corr = y + (h / 2) * (f(x, y) + f(x + h, y_pred))
        x = x + h
        y = y_corr
        steps.append((x, y))
        lines.append(f"{x:12.6f} {y_pred:14.8f} {y_corr:14.8f}")
    return steps, "\n".join(lines)


def safe_eval_ode(expr: str) -> Callable[[float, float], float]:
    """Evaluate dy/dx = f(x, y) from a math expression."""
    allowed = {
        "x": None,
        "y": None,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "exp": math.exp,
        "log": math.log,
        "sqrt": math.sqrt,
        "pi": math.pi,
        "e": math.e,
        "abs": abs,
        "pow": pow,
    }

    def f(x: float, y: float) -> float:
        allowed["x"] = x
        allowed["y"] = y
        return float(eval(expr, {"__builtins__": {}}, allowed))

    return f
