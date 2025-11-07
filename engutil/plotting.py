import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from matplotlib.patches import Circle
def init_latex():
    plt.rcParams.update({
        "text.usetex": True,
        "font.size": 14,
        "axes.labelsize": 14,
        "axes.titlesize": 16
    })

def _plot_core(data_list, legends=None, title="title", xlabel="Time [s]",
               ylabel="Amplitude", save_loc="def", xlim=None,ylim=None,
               plot_func=plt.plot, style_cycle=None, grid=None, xticks=None, yticks=None):
    """ wrapper function for plotting. a few examples:


        with xticks, yticks and grid=True
            t_q1 = np.array([0, 1, 1, 2, 2, 3, 3, 4, 4])
            y_q1 = np.array([1, 1, 0, 0, 0, 0, 0, 0, 1])  # PWM pattern

            engutil.plot_time_series([(t_q2, y_q2)], xticks=([0,1,2,3,4], ["0", "DT", "T", "T+DT", "2T"]), yticks=([0,1], ["0", "1"]), xlabel="Time", title="$Q_2$ PWM", grid=True)


    """
    init_latex()
    if legends is None:
        legends = [None] * len(data_list)
    
    plt.figure() # 
    style_iter = cycle(style_cycle) if style_cycle else None

    for (x, y), label in zip(data_list, legends):
        if plot_func is plt.stem:
            markerline, stemlines, baseline = plt.stem(x, y, label=label)
            if style_iter:
                c, _, _ = next(style_iter)
                plt.setp(markerline, color=c, markersize=4)
                plt.setp(stemlines, color=c, linewidth=0.8)
                plt.setp(baseline, color="k", linewidth=0.5, alpha=0.3)
        else:
            kwargs = {}
            if style_iter:
                c, ls, _ = next(style_iter)
                kwargs = dict(color=c, linestyle=ls)
            if label:
                plot_func(x, y, label=label, **kwargs)
            else:
                plot_func(x, y, **kwargs)


    plt.title(f"$\\textrm{{{title}}}$")
    plt.xlabel(f"$\\textrm{{{xlabel}}}$")
    plt.ylabel(f"$\\textrm{{{ylabel}}}$")
    plt.axis('tight')
    plt.tight_layout()
    
    def _wrap_labels(ticks):
        """Wrap string labels in $...$ for LaTeX rendering."""
        positions, labels = ticks
        wrapped = [f"$\\textrm{{{lbl}}}$" if isinstance(lbl, str) else lbl for lbl in labels]
        return positions, wrapped


    ## OPTIONAL ARGUMENTS ## 
    if xticks is not None:
        if isinstance(xticks, tuple) and len(xticks) == 2:
            plt.xticks(*_wrap_labels(xticks))
        else:
            plt.xticks(xticks)
    
    if yticks is not None:
        if isinstance(yticks, tuple) and len(yticks) == 2:
            plt.yticks(*_wrap_labels(yticks))
        else:
            plt.yticks(xticks)
    
    if any(lbl is not None for lbl in legends):
        plt.legend()

    if grid is not None:
        plt.grid()

    if xlim is not None:
        plt.xlim(xlim)
    
    if ylim is not None:
        plt.ylim(ylim)

    if save_loc is not None:
        # os.makedirs("figures", exist_ok=True)
        base = os.path.splitext(save_loc)[0]
        plt.savefig(f"{base}.svg")
        plt.savefig(f"{base}.png")

    plt.show()

def plot_time_series(data_list, legends=None, **kwargs):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    style_cycle = [(c, '-', None) for c in colors]
    _plot_core(data_list, legends=legends, plot_func=plt.plot,
               style_cycle=style_cycle, **kwargs)

def stem_time_series(data_list, legends=None, **kwargs):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    style_cycle = [(c, '-', None) for c in colors]
    _plot_core(data_list, legends=legends, plot_func=plt.stem,
               style_cycle=style_cycle, **kwargs)

def plot_real_phase(freq, Y, title="Spectrum Analysis", xlabel="Frequency [Hz]",
                      plot_type='plot', save_loc=None, grid=True, deg=True):
    """
    Plots the real part and phase of a complex signal in two separate subplots.

    Args:
        freq (array-like): The x-axis data (e.g., frequency).
        Y (array-like): The complex y-axis data (e.g., FFT output).
        title (str): The main title for the figure.
        xlabel (str): The label for the shared x-axis.
        plot_type (str): The type of plot to use, either 'plot' or 'stem'.
        save_loc (str, optional): File path to save the figure. Defaults to None.
        grid (bool): Whether to display a grid. Defaults to True.
        deg (bool): If True, plots phase in degrees. If False, in radians. Defaults to True.
    """
    init_latex()

    # Create a figure with two vertically stacked subplots that share an x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 7))
    fig.suptitle(f"$\\textrm{{{title}}}$", y=0.98)

    # --- 1. Real Part Subplot ---
    real_part = np.real(Y)
    if plot_type == 'stem':
        ax1.stem(freq, real_part)
    else:
        ax1.plot(freq, real_part)
    ax1.set_ylabel(r"$\textrm{Real Part}$")
    ax1.grid(grid)
    ax1.axis('tight')
    ax1.set_xscale('log')
    # --- 2. Phase Subplot ---
    if deg:
        phase = np.angle(Y, deg=True)
        phase_label = r"$\textrm{Phase [degrees]}$"
    else:
        phase = np.angle(Y)
        phase_label = r"$\textrm{Phase [radians]}$"

    if plot_type == 'stem':
        ax2.stem(freq, phase)
    else:
        ax2.plot(freq, phase)
    ax2.set_ylabel(phase_label)
    ax2.set_xlabel(f"$\\textrm{{{xlabel}}}$")
    ax2.grid(grid)
    ax2.axis('tight')
    ax2.set_xscale('log')

    # Use a tight layout to prevent labels from overlapping
    plt.tight_layout()
    
    # Adjust layout to make space for the main title
    fig.subplots_adjust(top=0.92)

    # Saving logic inspired by your _plot_core function
    if save_loc is not None:
        os.makedirs("figures", exist_ok=True)
        base = os.path.splitext(save_loc)[0]
        plt.savefig(f"figures/{base}.svg")
        plt.savefig(f"figures/{base}.png")

    plt.show()



# def plot_bode(freqs, responses,
#             title="Bode Plot",
#             xlabel="Frequency $f$ / Hz",
#             ylabel_left="Magnitude $\\left| H \\right|$ / dB re 1 V/V",
#             ylabel_right="Phase $\\angle H$ / $^\\circ$",
#             legends=None,
#             xlim=None,
#             save_loc=None,
#             return_fig=False):
#     """
#     Colorblind-friendly Bode plot using the Okabe–Ito palette.
#     Use: 
#     fig, ax1, ax2 = engutil.plot_bode(f_tweeter, [(t_mag, t_phase)], title="Tweeter", return_fig=True)
#     ax1.scatter(f0, H_max, color='k', marker='o')
#     ax1.legend(["t_mag", "scatter"])
#     ax2.legend(["phase"])
#     """
#     init_latex()
#     fig, ax1 = plt.subplots(figsize=(12,6))

#     # Okabe–Ito colorblind-safe palette
#     colors = [
#          "#E69F00", "#56B4E9", "#009E73", "#F0E442",
#         "#0072B2", "#D55E00", "#CC79A7","#999999"
#     ]

#     if legends is None:
#         legends = [f"Response {i+1}" for i in range(len(responses))]

#     # Left y-axis: magnitude
#     for i, (Z_mag, _) in enumerate(responses):
#         ax1.semilogx(freqs, Z_mag, color=colors[i % len(colors)],
#                      label=f"{legends[i]} $\\textrm{{(Mag)}}$")
#     ax1.set_xlabel(f"$\\textrm{{{xlabel}}}$")
#     ax1.set_ylabel(f"$\\textrm{{{ylabel_left}}}$", color='tab:blue')
#     ax1.tick_params(axis='y', labelcolor='tab:blue')
#     ax1.grid(True, which="both", ls="--", lw=0.7)

#     # Right y-axis: phase
#     ax2 = ax1.twinx()
#     phase_plotted = False
#     for i, (_, Z_phase) in enumerate(responses):
#         if isinstance(Z_phase, (np.ndarray, list)) and np.any(Z_phase):
#             ax2.semilogx(freqs, Z_phase, color=colors[i % len(colors)],
#                          linestyle="--", label=f"{legends[i]} $\\textrm{{(Phase)}}$")
#             phase_plotted = True

#     if phase_plotted:
#         ax2.set_ylabel(f"$\\textrm{{{ylabel_right}}}$", color='tab:red')
#         ax2.tick_params(axis='y', labelcolor='tab:red')
#     else:
#         fig.delaxes(ax2)

#     # Combined legend
#     lines_1, labels_1 = ax1.get_legend_handles_labels()
#     lines_2, labels_2 = ax2.get_legend_handles_labels()
#     labels_all = [f"$\\textrm{{{lbl}}}$" for lbl in (labels_1 + labels_2)]
#     ax1.legend(lines_1 + lines_2, labels_all, loc='best')

#     plt.title(f"$\\textrm{{{title}}}$")
    
#     if xlim is not None:
#         plt.xlim(xlim)
#     plt.tight_layout()
#     if save_loc:
#         plt.savefig(f"{save_loc}.png", bbox_inches="tight")
#         plt.savefig(f"{save_loc}.svg", bbox_inches="tight")
#     if return_fig:
#         return fig, ax1, ax2, plt if phase_plotted else None
#     plt.show()

def plot_bode(freqs, responses,
            title="Bode Plot",
            xlabel="Frequency $f$ / Hz",
            ylabel_left="Magnitude $\\left| H \\right|$ / dB re 1 V/V",
            ylabel_right="Phase $\\angle H$ / $^\\circ$",
            legends=None,
            xlim=None,
            save_loc=None,
            return_fig=False):

    init_latex()
    fig, ax1 = plt.subplots(figsize=(12,6))

    color_left = 'tab:blue'
    color_right = 'tab:red'

    if legends is None:
        legends = [f"\\textrm{{Response}} ${i+1}$" for i in range(len(responses))]

    # Left y-axis: magnitude
    for i, (Z_mag, _) in enumerate(responses):
        ax1.semilogx(freqs, Z_mag, color=color_left,
                     label=f"{legends[i]} $\\textrm{{(Mag)}}$")
    ax1.set_xlabel(f"$\\textrm{{{xlabel}}}$")
    ax1.set_ylabel(f"$\\textrm{{{ylabel_left}}}$", color=color_left)
    ax1.tick_params(axis='y', labelcolor=color_left)
    ax1.grid(True, which="both", ls="--", lw=0.7)

    # Right y-axis: phase
    ax2 = ax1.twinx()
    phase_plotted = False
    for i, (_, Z_phase) in enumerate(responses):
        if isinstance(Z_phase, (np.ndarray, list)) and np.any(Z_phase):
            ax2.semilogx(freqs, Z_phase, color=color_right,
                         linestyle="--", label=f"{legends[i]} $\\textrm{{(Phase)}}$")
            phase_plotted = True

    if phase_plotted:
        ax2.set_ylabel(f"$\\textrm{{{ylabel_right}}}$", color=color_right)
        ax2.tick_params(axis='y', labelcolor=color_right)
    else:
        fig.delaxes(ax2)

    # Combined legend
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    labels_all = [f"$\\textrm{{{lbl}}}$" for lbl in (labels_1 + labels_2)]
    ax1.legend(lines_1 + lines_2, labels_all, loc='best')

    plt.title(f"$\\textrm{{{title}}}$")
    
    if xlim is not None:
        plt.xlim(xlim)
    plt.tight_layout()
    if save_loc:
        plt.savefig(f"{save_loc}.png", bbox_inches="tight")
        plt.savefig(f"{save_loc}.svg", bbox_inches="tight")
    if return_fig:
        return fig, ax1, ax2, plt if phase_plotted else None
    plt.show()


def read_ltspice_export(file_path):
    """Read LTSpice exported AC analysis file."""
    with open(file_path, 'r', encoding='ISO-8859-1') as file:
        lines = file.readlines()
    
    freqs, mags, phases = [], [], []
    for line in lines[1:]:
        parts = line.strip().split('\t')
        freq = float(parts[0])
        val = parts[1][1:-1]  # "(mag dB, phase °)"
        mag, phase = val.split(',')
        mags.append(float(mag[:-2]))
        phases.append(float(phase[:-1]))
        freqs.append(freq)

    return (np.array(freqs),
            np.array(mags),
            np.unwrap(np.deg2rad(phases))*180/np.pi)

def plot_ltspice(
        file_paths, 
        legends=None, 
        title="Bode Plot", 
        save_loc=None,
        xlabel="Frequency $f$ / Hz",
        ylabel_left="Magnitude [dB]",
        ylabel_right="Phase [deg]",
        xlim=None):
    """
    Wrapper: reads one or more LTSpice exported text files
    and forwards data into plot_bode().
    """
    init_latex()
    if legends is None:
        legends = [f"File {i+1}" for i in range(len(file_paths))]

    responses = []
    freqs = None

    for file_path in file_paths:
        f, mags, phases = read_ltspice_export(file_path)
        if freqs is None:
            freqs = f
        responses.append((mags, phases))

    plot_bode(freqs,
              responses,
              legends=legends,
              title=title,
              xlabel=xlabel,
              ylabel_left=ylabel_left,
              ylabel_right=ylabel_right,
              xlim=xlim,
              save_loc=save_loc)




def plot_zplane(b, a, title="Pole-Zero plot", xlabel="Real Part", ylabel="Imaginary Part", save_loc=None):
    
    init_latex()
    zeros = np.roots(b)
    poles = np.roots(a)

    fig, ax = plt.subplots()

    unit_circle = Circle((0, 0), radius=1, fill=False, color='gray', ls='--', alpha=0.6)
    ax.add_patch(unit_circle)

    ax.plot(np.real(zeros), np.imag(zeros), 'o', markersize=8, label="$\\textrm{Zeros}$")
    ax.plot(np.real(poles), np.imag(poles), 'x', markersize=8, label='Poles')

    # ax.set_title('Pole-Zero Plot')
    # ax.set_xlabel('Real Part')
    # ax.set_ylabel("$")

    ax.set_title(f"$\\textrm{{{title}}}$")
    ax.set_xlabel(f"$\\textrm{{{xlabel}}}$")
    ax.set_ylabel(f"$\\textrm{{{ylabel}}}$")

    ax.axvline(0, color='gray', lw=0.5)
    ax.axhline(0, color='gray', lw=0.5)
    
    ax.grid(True)
    ax.set_aspect('equal')
    ax.legend()


    if save_loc:
        plt.savefig(f"{save_loc}.png", bbox_inches="tight")
        plt.savefig(f"{save_loc}.svg", bbox_inches="tight")
    plt.show()
    plt.show()
