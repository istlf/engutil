import sys
import importlib

from .signals import (
    generate_sine,
    generate_square,
    generate_ramp,
    generate_time,
    make_spectrum
)

from .util import (

)

from .plotting import (
    init_latex,
    plot_time_series,
    stem_time_series,
    plot_real_phase,
    plot_bode,
    plot_ltspice
)

def reload_self():
    """Reload engutil and all its submodules (for Jupyter dev use)."""
    modules = [m for m in sys.modules if m.startswith("engutil")]
    for m in modules:
        importlib.reload(sys.modules[m])
    import engutil
    return engutil

__all__ = [
    "generate_sine",
    "generate_square",
    "generate_ramp",
    "generate_time",
    "init_latex",
    "plot_time_series",
    "stem_time_series",
    "plot_real_phase",
    "plot_bode",
    "plot_ltspice"
    "make_spectrum",
    "reload_self"
    
]
