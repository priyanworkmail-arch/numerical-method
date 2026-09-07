"""Conversational handler for numerical methods chatbot."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto

from numerical_methods import (
    lagrange_interpolation,
    modified_euler,
    newton_backward_interpolation,
    newton_forward_interpolation,
    parse_number_list,
    parse_points,
    runge_kutta_4,
    safe_eval_ode,
)


class Method(Enum):
    NONE = auto()
    LAGRANGE = auto()
    FORWARD = auto()
    BACKWARD = auto()
    RUNGE_KUTTA = auto()
    MODIFIED_EULER = auto()


@dataclass
class Session:
    method: Method = Method.NONE
    step: int = 0
    data: dict = field(default_factory=dict)


def detect_method(text: str) -> Method | None:
    t = text.lower()
    if any(k in t for k in ("lagrange", "lagrangian")):
        return Method.LAGRANGE
    if "forward" in t and "interp" in t:
        return Method.FORWARD
    if "backward" in t and "interp" in t:
        return Method.BACKWARD
    if any(k in t for k in ("runge", "rk4", "rk-4", "runge-kutta", "runge kutta")):
        return Method.RUNGE_KUTTA
    if any(k in t for k in ("modified euler", "mod euler", "euler modified", "heun")):
        return Method.MODIFIED_EULER
    if "forward" in t and "difference" in t:
        return Method.FORWARD
    if "backward" in t and "difference" in t:
        return Method.BACKWARD
    return None


HELP_TEXT = """
**Numerical Methods Chatbot**

Choose a method or type its name:

1. **Lagrange Interpolation** — unequally spaced points
2. **Forward Interpolation** — Newton forward (equally spaced)
3. **Backward Interpolation** — Newton backward (equally spaced)
4. **Runge-Kutta (RK4)** — solve ODE dy/dx = f(x,y)
5. **Modified Euler** — improved Euler / Heun method for ODEs

Type `help` anytime. Type `reset` to start over.
Type `menu` to see options again.
""".strip()


def start_method(method: Method) -> tuple[str, Session]:
    session = Session(method=method, step=1)
    prompts = {
        Method.LAGRANGE: (
            "**Lagrange Interpolation**\n\n"
            "Send your data points, then the x value to interpolate.\n\n"
            "Format:\n"
            "`x: 0, 1, 2, 3`\n"
            "`y: 1, 2, 5, 10`\n"
            "Or pairs: `0,1; 1,2; 2,5; 3,10`\n\n"
            "Then on the next message send: `x = 1.5`"
        ),
        Method.FORWARD: (
            "**Newton Forward Interpolation**\n\n"
            "Send equally spaced points:\n"
            "`x: 0, 1, 2, 3, 4`\n"
            "`y: 1, 8, 27, 64, 125`\n\n"
            "Then send the x value: `x = 2.5`"
        ),
        Method.BACKWARD: (
            "**Newton Backward Interpolation**\n\n"
            "Send equally spaced points:\n"
            "`x: 0, 1, 2, 3, 4`\n"
            "`y: 1, 8, 27, 64, 125`\n\n"
            "Then send the x value: `x = 3.5`"
        ),
        Method.RUNGE_KUTTA: (
            "**Runge-Kutta 4th Order (RK4)**\n\n"
            "Send ODE and initial values in one message:\n\n"
            "`dy/dx = x + y`\n"
            "`x0 = 0, y0 = 1, h = 0.1, x_end = 1`\n\n"
            "Use Python math: sin, cos, exp, log, sqrt, pi, e"
        ),
        Method.MODIFIED_EULER: (
            "**Modified Euler Method**\n\n"
            "Send ODE and initial values:\n\n"
            "`dy/dx = x - y`\n"
            "`x0 = 0, y0 = 1, h = 0.1, x_end = 1`"
        ),
    }
    return prompts[method], session


def _parse_x_value(text: str) -> float | None:
    m = re.search(r"x\s*=\s*([\d.\-eE+]+)", text, re.I)
    if m:
        return float(m.group(1))
    try:
        nums = parse_number_list(text)
    except ValueError:
        return None
    if len(nums) == 1:
        return nums[0]
    return None


def _parse_ode_params(text: str) -> dict:
    expr_match = re.search(
        r"(?:dy/dx|dy/dt|y'|f\(x,y\))\s*=\s*(.+?)(?:\n|$|x0)",
        text,
        re.I | re.S,
    )
    if not expr_match:
        expr_match = re.search(r"=\s*(.+?)(?:\n|$)", text)
    if not expr_match:
        raise ValueError("Could not find ODE. Example: dy/dx = x + y")

    expr = expr_match.group(1).strip().rstrip(",")

    def grab(name: str) -> float | None:
        m = re.search(rf"{name}\s*=\s*([\d.\-eE+]+)", text, re.I)
        return float(m.group(1)) if m else None

    x0 = grab("x0")
    y0 = grab("y0")
    h = grab("h")
    x_end = grab("x_end") or grab("xn") or grab("xend")

    if None in (x0, y0, h, x_end):
        raise ValueError("Need x0, y0, h, and x_end. Example: x0=0, y0=1, h=0.1, x_end=1")

    return {"expr": expr, "x0": x0, "y0": y0, "h": h, "x_end": x_end}


def _try_one_shot(method: Method, text: str) -> tuple[str, Session] | None:
    """If the message that triggered this method already contains the data
    needed to solve it (points and/or an x value, or a full ODE spec),
    skip the step-by-step prompts and solve directly.

    Returns None when the message doesn't contain usable data, so the
    caller falls back to the normal guided, step-by-step flow.
    """
    if method in (Method.LAGRANGE, Method.FORWARD, Method.BACKWARD):
        try:
            xs, ys = parse_points(text)
        except Exception:
            return None
        if len(xs) < 2:
            return None

        x_val = _parse_x_value(text)
        if x_val is None:
            # Points found but no target x yet -- jump straight to step 2.
            session = Session(method=method, step=2, data={"xs": xs, "ys": ys})
            return (
                f"Got {len(xs)} points.\n"
                f"x = {xs}\n"
                f"y = {ys}\n\n"
                "Now send the x value to interpolate (e.g. `x = 2.5`).",
                session,
            )

        try:
            if method == Method.LAGRANGE:
                result = lagrange_interpolation(xs, ys, x_val)
                detail = ""
            elif method == Method.FORWARD:
                result, detail = newton_forward_interpolation(xs, ys, x_val)
            else:
                result, detail = newton_backward_interpolation(xs, ys, x_val)
        except Exception as exc:
            fallback = Session(method=method, step=2, data={"xs": xs, "ys": ys})
            return f"Error: {exc}\n\nTry again or type `reset`.", fallback

        name = method.name.replace("_", " ").title()
        msg = f"**{name} Result**\n\nf({x_val}) ≈ **{result:.8g}**"
        if detail:
            msg += f"\n\nSteps:\n```\n{detail}\n```"
        msg += "\n\nType another method name or `reset`."
        return msg, Session()

    if method in (Method.RUNGE_KUTTA, Method.MODIFIED_EULER):
        try:
            params = _parse_ode_params(text)
        except Exception:
            return None

        try:
            f = safe_eval_ode(params["expr"])
            if method == Method.RUNGE_KUTTA:
                steps, table = runge_kutta_4(
                    f, params["x0"], params["y0"], params["h"], params["x_end"]
                )
                title = "Runge-Kutta 4"
            else:
                steps, table = modified_euler(
                    f, params["x0"], params["y0"], params["h"], params["x_end"]
                )
                title = "Modified Euler"
        except Exception as exc:
            return f"Error: {exc}\n\nTry again or type `reset`.", Session()

        final_x, final_y = steps[-1]
        msg = (
            f"**{title}**\n\n"
            f"ODE: dy/dx = {params['expr']}\n"
            f"x0={params['x0']}, y0={params['y0']}, h={params['h']}, "
            f"x_end={params['x_end']}\n\n"
            f"**Final: y({final_x:.6g}) ≈ {final_y:.8g}**\n\n"
            f"Iteration table:\n```\n{table}\n```\n\n"
            "Type another method or `reset`."
        )
        return msg, Session()

    return None


def handle_message(text: str, session: Session | None) -> tuple[str, Session]:
    text = text.strip()
    if not text:
        return "Please type a message.", session or Session()

    lower = text.lower()
    if lower in ("help", "menu", "start", "hi", "hello"):
        return HELP_TEXT, Session()

    if lower in ("reset", "cancel", "new"):
        return "Session reset. " + HELP_TEXT, Session()

    if session is None or session.method == Method.NONE:
        method = detect_method(text)
        if method is None:
            if lower.isdigit() and 1 <= int(lower) <= 5:
                method = list(Method)[int(lower)]
            else:
                return (
                    "I can solve numerical methods problems.\n\n" + HELP_TEXT,
                    Session(),
                )

        one_shot = _try_one_shot(method, text)
        if one_shot is not None:
            return one_shot

        return start_method(method)

    # Active session
    try:
        if session.method in (Method.LAGRANGE, Method.FORWARD, Method.BACKWARD):
            if session.step == 1:
                xs, ys = parse_points(text)
                session.data["xs"] = xs
                session.data["ys"] = ys
                session.step = 2
                return (
                    f"Got {len(xs)} points.\n"
                    f"x = {xs}\n"
                    f"y = {ys}\n\n"
                    "Now send the x value to interpolate (e.g. `x = 2.5`).",
                    session,
                )

            x_val = _parse_x_value(text)
            if x_val is None:
                return "Send a single x value, e.g. `x = 1.5`", session

            xs = session.data["xs"]
            ys = session.data["ys"]

            if session.method == Method.LAGRANGE:
                result = lagrange_interpolation(xs, ys, x_val)
                detail = ""
            elif session.method == Method.FORWARD:
                result, detail = newton_forward_interpolation(xs, ys, x_val)
            else:
                result, detail = newton_backward_interpolation(xs, ys, x_val)

            name = session.method.name.replace("_", " ").title()
            msg = f"**{name} Result**\n\nf({x_val}) ≈ **{result:.8g}**"
            if detail:
                msg += f"\n\nSteps:\n```\n{detail}\n```"
            msg += "\n\nType another method name or `reset`."
            return msg, Session()

        if session.method in (Method.RUNGE_KUTTA, Method.MODIFIED_EULER):
            params = _parse_ode_params(text)
            f = safe_eval_ode(params["expr"])
            if session.method == Method.RUNGE_KUTTA:
                steps, table = runge_kutta_4(
                    f, params["x0"], params["y0"], params["h"], params["x_end"]
                )
                title = "Runge-Kutta 4"
            else:
                steps, table = modified_euler(
                    f, params["x0"], params["y0"], params["h"], params["x_end"]
                )
                title = "Modified Euler"

            final_x, final_y = steps[-1]
            msg = (
                f"**{title}**\n\n"
                f"ODE: dy/dx = {params['expr']}\n"
                f"x0={params['x0']}, y0={params['y0']}, h={params['h']}, "
                f"x_end={params['x_end']}\n\n"
                f"**Final: y({final_x:.6g}) ≈ {final_y:.8g}**\n\n"
                f"Iteration table:\n```\n{table}\n```\n\n"
                "Type another method or `reset`."
            )
            return msg, Session()

    except Exception as exc:
        return f"Error: {exc}\n\nTry again or type `reset`.", session

    return HELP_TEXT, Session()
