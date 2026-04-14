import sys
import importlib

from .signals import (
    generate_sine,
    generate_square,
    generate_ramp,
    generate_time,
    make_spectrum
)


from .plotting import (
    init_latex,
    plot_time_series,
    stem_time_series,
    plot_real_phase,
    plot_bode,
    plot_ltspice,
    plot_zplane,
    read_ltspice_export
)

from .acoustics import (
    radiation_impedance_piston_in_baffle
)
from .rf import (
    TwoPortNetwork,
    to_cartesian,
    to_polar,
    calc_transducer_gain,
    noise_figure_circle,
    available_gain_circle,
    reflection_2_impedance
)
from .util import (
    load_complex_csv,
    open_csv,
    read_bode_csv,
    tf_to_magphase,
    find_max,
    find_min,
    find_f1_f2,
    round_to_E,
    pprint,
    write_ltspice_params,
    percent,
    mag2db,
    db2mag,
    pol2cart,
    cart2pol,
    load_ads_csv,
    generate_noise_figure_latex_table,
    generate_available_gain_latex_table,
    to_db_pwr,
    generate_circle_locus,
    to_linear,
    save_points_as_dat
)

from .latex import (
    create_smith_chart_tex,
    append_point_to_tex,
    append_circle_to_tex
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
    "read_ltspice_export",
    "plot_ltspice"
    "make_spectrum",
    "reload_self",
    "plot_zplane",

    "radiation_impedance_piston_in_baffle",

    "TwoPortNetwork",
    "to_cartesian",
    "to_polar",
    "calc_transducer_gain",

    
    "load_complex_csv",
    "open_csv",
    "read_bode_csv",
    "tf_to_magphase",
    "find_max",
    "find_min",
    "find_f1_f2",
    "round_to_E",
    "pprint",
    "write_ltspice_params",
    "percent",
    "mag2db",
    "db2mag",
    "cart2pol",
    "pol2cart",
    "load_ads_csv"

]
