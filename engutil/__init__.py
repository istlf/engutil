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
    CB_COLORS,
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
    calc_transducer_gain,
    reflection_2_impedance,
    impedance_2_reflection,
    L_LP_to_BP,
    C_LP_to_BP,
    design_coupled_line_filter,
    design_lumped_bandpass,
    transform_shunt_element,
    transform_series_element,
    parse_ads_data,
    plot_mag,
    conductance_matrix,
    calculate_vswr_out,
    conversion_gain,
    conductance_matrix,
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
    db2pow,
    pow2db,
    watt2dbm,
    dbm2watt,
    pol2cart,
    cart2pol,
    load_ads_csv,
    to_db_pwr,
    to_linear,
    save_points_as_dat,
    to_cartesian,
    to_polar,
    to_db_pwr,
    np2db,
    db2np

)

from .latex import (
    create_smith_chart_tex,
    append_point_to_tex,
    append_circle_to_tex,
    generate_noise_figure_latex_table,
    generate_available_gain_latex_table,
    to_latex,
    generate_filter_component_latex_table,
    append_string_to_tex,
    generate_latex_table_coupled_line,
    generate_tline_latex_table,
    gen_latex_command
)

from .rf_plot import (
    parse_ads_data,
    plot_mag,
    get_bandpass_stats,
    get_filter_bounds
)

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
    "plot_ltspice",
    "make_spectrum",
    "reload_self",
    "plot_zplane",

    "radiation_impedance_piston_in_baffle",

    "TwoPortNetwork",
    "to_cartesian",
    "to_polar",
    "conversion_gain",
    "conductance_matrix",
    
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
    "watt2dbm",
    "dbm2watt",
    
    "load_ads_csv",
    "design_coupled_line_filter",
    "design_lumped_bandpass",
    "transform_shunt_element",
    "transform_series_element",
    "to_latex"
]
