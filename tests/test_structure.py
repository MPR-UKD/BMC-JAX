"""Rules about the SHAPE of the package, checked mechanically rather than by good intentions.

"Modular and well organised" is not a feeling. These are the properties that make it true, read
off the source with ast so they cannot drift: nothing here imports the package, so a rule cannot
be satisfied by accident at runtime.

The load-bearing one is `test_optimize_does_not_import_the_solver`. A bounded Levenberg-Marquardt
has nothing to do with Bloch-McConnell; if it reaches for the solver then the split was
cosmetic and `lm_minimize` is not usable on its own.
"""
import ast
from pathlib import Path

import pytest

SRC = Path(__file__).parents[1] / "src" / "bmc_jax"

#: Which modules each layer is allowed to import FROM THIS PACKAGE. Layers only ever depend
#: downward, so the graph is a DAG and there is a bottom to start reading at.
ALLOWED = {
    "precision":        set(),
    "io.config":              {"io.parameters"},
    "io.parameters.pool":     set(),
    "io.parameters.water_pool":  {"io.parameters.pool"},
    "io.parameters.cest_pool":   {"io.parameters.pool"},
    "io.parameters.mt_pool":     {"io.parameters.cest_pool"},
    "io.parameters.system":   set(),
    "io.parameters.options":  set(),
    "io.parameters.parameters": {"io.parameters.cest_pool", "io.parameters.mt_pool",
                                 "io.parameters.options", "io.parameters.system",
                                 "io.parameters.water_pool"},
    "fitting.problem":  {"jax_solver"},
    "fitting.spec":     {"jax_solver"},
    "fitting.optimize": set(),                                   # <- the important one
    "fitting.result":   {"fitting.spec"},
    "fitting.residual": {"jax_solver", "fitting.spec"},
    "fitting.batching": {"jax_solver", "fitting.spec", "fitting.residual",
                         "fitting.optimize", "fitting.problem", "fitting.result"},
}


def module_path(dotted: str) -> Path:
    return SRC.joinpath(*dotted.split(".")).with_suffix(".py")


def internal_imports(path: Path) -> set[str]:
    """Every bmc_jax module this file imports, as dotted names relative to the package."""
    tree = ast.parse(path.read_text())
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level:           # from .x / from ..x import
            here = path.relative_to(SRC).with_suffix("").parts[:-1]
            base = list(here[: len(here) - (node.level - 1)]) if node.level > 1 else list(here)
            mod = (node.module or "").split(".")
            out.add(".".join([*base, *[m for m in mod if m]]))
        elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith("bmc_jax"):
            out.add(node.module[len("bmc_jax."):])
    return {m for m in out if m}


@pytest.mark.parametrize("dotted", sorted(ALLOWED))
def test_import_layering(dotted):
    """Each module imports only from the layers below it -- so the graph is a DAG."""
    got = internal_imports(module_path(dotted))
    # a dependency on 'jax_solver.simulator' satisfies an allowance of 'jax_solver'
    for dep in got:
        assert any(dep == a or dep.startswith(a + ".") for a in ALLOWED[dotted]), (
            f"{dotted} imports {dep}, which is not in its allowed layer {ALLOWED[dotted] or '{}'}")


def test_optimize_does_not_import_the_solver():
    """The optimiser must be usable without any of the physics. This is the whole point."""
    src = module_path("fitting.optimize").read_text()
    assert "jax_solver" not in src, "optimize.py reached for the solver; the split is cosmetic"
    assert internal_imports(module_path("fitting.optimize")) == set()


def test_no_environment_variables_anywhere_in_the_package():
    """An installed library whose behaviour depends on undocumented env vars is not shippable.

    The research tree it came from reads a dozen FIT_* variables at import time. None of that
    may follow the code in here: configuration arrives as arguments.

    Checked with ast rather than by grepping for the text, because a docstring that SHOWS the
    caller how to do it themselves -- as precision.py does -- is not an environment read.
    """
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            # os.environ[...] / os.environ.get(...) / os.getenv(...)
            hit = (isinstance(node, ast.Attribute)
                   and node.attr in ("environ", "getenv")
                   and isinstance(node.value, ast.Name) and node.value.id == "os")
            if hit:
                offenders.append(f"{path.relative_to(SRC)}:{node.lineno}")
    assert offenders == [], f"environment reads found at {offenders}"


def test_public_api_has_no_private_names():
    """__all__ is the contract. A leading underscore in it means the boundary is wrong."""
    import bmc_jax.fitting as F
    import bmc_jax.io as IO
    for mod in (F, IO):
        assert hasattr(mod, "__all__"), f"{mod.__name__} has no __all__"
        bad = [n for n in mod.__all__ if n.startswith("_")]
        assert bad == [], f"{mod.__name__}.__all__ exports private names: {bad}"


def test_every_exported_name_exists():
    import bmc_jax.fitting as F
    import bmc_jax.io as IO
    import bmc_jax.jax_solver as S
    for mod in (F, IO, S):
        missing = [n for n in mod.__all__ if not hasattr(mod, n)]
        assert missing == [], f"{mod.__name__}.__all__ promises {missing}, which do not exist"


def test_parameter_objects_are_immutable():
    """Validation runs once, at construction. Freezing is what makes once enough.

    Without it, `pool.f = 2.0` after construction would silently succeed and the object would be
    in a state nothing ever checked. bmctool buys the same guarantee with a property setter per
    field; this costs one word per class.
    """
    from dataclasses import FrozenInstanceError, replace
    from bmc_jax.io import example_config
    p = example_config("amide")
    for obj, attr in ((p.water_pool, "f"), (p.cest_pools[0], "k"), (p.system, "b0"),
                      (p.options, "scale"), (p, "mt_pool")):
        with pytest.raises(FrozenInstanceError):
            setattr(obj, attr, 0.5)
    # and building a variant re-runs every check
    assert replace(p.cest_pools[0], k=200.0).k == 200.0
    with pytest.raises(ValueError):
        replace(p.cest_pools[0], k=-1.0)


def test_the_old_names_still_work():
    """The split must not break a single caller. Every pre-split name is still importable."""
    from bmc_jax.fitting import (FitSpec, Slot, apply_x, fit_phantom_adam,
                                 fit_phantom_lm, fit_phantom_lm_sharded, fit_voxel,
                                 make_phantom_residual, make_single_residual)
    from bmc_jax.fitting import _lm_one, lm_minimize
    assert _lm_one is lm_minimize, "the compatibility alias drifted from the public name"
