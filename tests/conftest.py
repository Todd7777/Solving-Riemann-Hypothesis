import sys, pathlib
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(autouse=True)
def _flint_context_guard():
    """python-flint's precision (ctx.dps/ctx.prec) and series cap are global
    state; save and restore them around every test so that one module's
    settings cannot leak into another's reference computations."""
    from flint import ctx
    dps, cap = ctx.dps, ctx.cap
    try:
        yield
    finally:
        ctx.dps, ctx.cap = dps, cap
