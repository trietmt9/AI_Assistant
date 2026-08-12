"""`calc` and `run_python` — the computation tools.

CLAUDE.md design rule 3 and PLAN.md §7.3: **all computation goes through `calc`.**
No model at this scale does reliable arithmetic, and a bigger model buys better
problem *setup*, not a calculator. The model's job is to state the problem; SymPy's
job is to solve it.

`run_python` is the escape hatch for anything SymPy cannot express — numerics,
signal processing, plotting. It is sandboxed per CLAUDE.md: subprocess, timeout,
no network, scratch cwd.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import textwrap
from dataclasses import dataclass

from pydantic_ai import ModelRetry, ToolFailed

from assistant.config import settings

log = logging.getLogger(__name__)

CALC_TIMEOUT_S = 20
PYTHON_TIMEOUT_S = 30
MAX_OUTPUT_CHARS = 4000


# --- calc ----------------------------------------------------------------


def _sympy_namespace() -> dict:
    """A curated SymPy namespace.

    `sympify` on raw input is `eval` in a trenchcoat. `parse_expr` with an
    explicit `local_dict` and no builtins is the supported way to keep that
    bounded, so the model can call `integrate(...)` without being able to call
    `__import__(...)`.
    """
    import sympy as sp

    names = (
        # calculus / algebra
        "integrate diff limit series solve solveset simplify expand factor "
        "collect cancel apart together nsimplify radsimp trigsimp powsimp "
        "summation product Sum Product Derivative Integral Limit "
        # equations & matrices
        "Eq Matrix eye zeros ones det transpose "
        # functions
        "sin cos tan asin acos atan atan2 sinh cosh tanh exp log ln sqrt Abs "
        "floor ceiling factorial binomial gamma erf "
        # constants & types
        "pi E I oo nan Rational Integer Float Symbol symbols "
        # numeric evaluation
        "N evalf re im conjugate arg "
        # misc
        "linsolve nonlinsolve dsolve laplace_transform fourier_transform "
        "inverse_laplace_transform Heaviside DiracDelta"
    ).split()

    ns = {n: getattr(sp, n) for n in names if hasattr(sp, n)}
    # Bare single letters are overwhelmingly intended as symbols.
    for letter in "abcdefghijklmnopqrstuvwxyz":
        ns.setdefault(letter, sp.Symbol(letter))
    for letter in "ABCDFGHJKLMNOPQRSTUVWXYZ":  # E and I are constants above
        ns.setdefault(letter, sp.Symbol(letter))
    return ns


@dataclass(slots=True)
class CalcResult:
    expression: str
    result: str
    pretty: str = ""
    numeric: str = ""

    def render(self) -> str:
        out = [f"{self.expression} = {self.result}"]
        if self.numeric and self.numeric != self.result:
            out.append(f"numeric: {self.numeric}")
        return "\n".join(out)


def _convert_units(expression: str) -> CalcResult | None:
    """Handle `3.5 eV to J` style conversions with pint.

    Returns None when the expression is not a conversion, so `calc` falls
    through to SymPy. PLAN.md §7.3 lists units as part of this tool rather than a
    separate one — the ~8-tool budget in §9 is tight enough without splitting it.
    """
    for sep in (" to ", " in ", " -> ", " into "):
        if sep not in expression:
            continue
        left, right = expression.rsplit(sep, 1)
        try:
            import pint

            ureg = pint.UnitRegistry()
            quantity = ureg.Quantity(left.strip())
            converted = quantity.to(right.strip())
        except Exception:
            return None  # not a unit conversion after all
        return CalcResult(
            expression=expression,
            result=f"{converted:~P}",
            numeric=f"{converted.magnitude:.10g} {converted.units:~P}",
        )
    return None


def calc(expression: str) -> str:
    """Evaluate or manipulate a mathematical expression with SymPy.

    Use this for ALL arithmetic, algebra, calculus and unit conversion — never
    compute in your head. Accepts SymPy call syntax, so
    `integrate(x**2*sin(x), x)`, `solve(x**2 - 5*x + 6, x)` and
    `diff(exp(2*x)*cos(x), x)` all work, as does `3.5 eV to J`.

    Args:
        expression: The expression to evaluate.
    """
    expression = (expression or "").strip()
    if not expression:
        raise ModelRetry("`expression` was empty. Pass something to evaluate.")

    if (converted := _convert_units(expression)) is not None:
        return converted.render()

    try:
        import sympy as sp
        from sympy.parsing.sympy_parser import (
            convert_xor,
            implicit_multiplication_application,
            parse_expr,
            standard_transformations,
        )
    except ImportError as exc:  # pragma: no cover
        raise ToolFailed(f"SymPy unavailable: {exc}") from exc

    transformations = standard_transformations + (
        implicit_multiplication_application,
        convert_xor,  # let `x^2` mean `x**2`, which models write constantly
    )

    try:
        expr = parse_expr(
            expression,
            local_dict=_sympy_namespace(),
            transformations=transformations,
            global_dict={},  # no builtins
            evaluate=True,
        )
    except Exception as exc:
        # A parse failure is the model's mistake and it can fix it -- give it the
        # error rather than crashing the run (CLAUDE.md: return validation errors
        # to the model to retry).
        raise ModelRetry(
            f"Could not parse {expression!r}: {type(exc).__name__}: {exc}. "
            "Use SymPy syntax, e.g. integrate(x**2*sin(x), x) or solve(x**2-4, x)."
        ) from exc

    try:
        simplified = sp.simplify(expr) if expr.free_symbols else expr
    except Exception:
        simplified = expr

    result = CalcResult(expression=expression, result=str(simplified))
    try:
        evaluated = sp.N(simplified, 12)
        if evaluated.is_number:
            result.numeric = str(evaluated)
    except Exception:
        pass

    try:
        result.pretty = sp.pretty(simplified, use_unicode=True)
    except Exception:
        pass

    return result.render()


# --- run_python ----------------------------------------------------------

_PREAMBLE = """\
import sys, math, json
try:
    import numpy as np
except ImportError:
    np = None
"""


def _sandbox_argv(script_path: str) -> tuple[list[str], bool]:
    """Build the subprocess argv, isolating the network where possible.

    `unshare -rn` gives an unprivileged network namespace with no interfaces,
    which is real isolation rather than an honour system. It is not available
    everywhere (some hardened kernels disable unprivileged user namespaces), so
    the caller is told which mode it got instead of being quietly downgraded.
    """
    base = [sys.executable, script_path]
    if shutil.which("unshare"):
        probe = subprocess.run(
            ["unshare", "-rn", "true"], capture_output=True, timeout=10
        )
        if probe.returncode == 0:
            return ["unshare", "-rn", *base], True
    return base, False


def run_python(code: str) -> str:
    """Execute Python in a sandboxed subprocess and return its stdout.

    Use for numerics, signal processing and plotting — anything `calc` cannot
    express symbolically. NumPy is preloaded as `np` when available. Print what
    you want to see; the return value of the last expression is not echoed.

    The process has a timeout, a scratch working directory, and no network.

    Args:
        code: Python source to execute.
    """
    code = (code or "").strip()
    if not code:
        raise ModelRetry("`code` was empty. Pass a Python snippet to execute.")

    scratch = settings.scratch_dir
    scratch.mkdir(parents=True, exist_ok=True)
    script = scratch / "_run.py"
    script.write_text(_PREAMBLE + "\n" + textwrap.dedent(code))

    argv, isolated = _sandbox_argv(str(script))
    if not isolated:
        log.warning("network isolation unavailable; run_python is NOT network-sandboxed")

    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(scratch),
        "PYTHONDONTWRITEBYTECODE": "1",
        "MPLBACKEND": "Agg",  # plotting must not try to open a window
        "OPENBLAS_NUM_THREADS": "4",
    }

    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=PYTHON_TIMEOUT_S,
            cwd=scratch,
            env=env,
        )
    except subprocess.TimeoutExpired:
        raise ToolFailed(
            f"Execution exceeded {PYTHON_TIMEOUT_S}s and was killed. "
            "Reduce the work or vectorise it."
        ) from None
    except OSError as exc:
        raise ToolFailed(f"Could not start the sandbox: {exc}") from exc

    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()

    if proc.returncode != 0:
        # A traceback is something the model can act on, so this is a retry
        # rather than a hard failure.
        raise ModelRetry(f"Script exited {proc.returncode}:\n{stderr[-1500:]}")

    if not stdout:
        return "(script produced no output — remember to print() your results)"

    if len(stdout) > MAX_OUTPUT_CHARS:
        return stdout[:MAX_OUTPUT_CHARS] + f"\n... truncated ({len(stdout)} chars total)"
    return stdout


__all__ = ["calc", "run_python"]
