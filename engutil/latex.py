import numpy as np
import os
import re
import math

def create_smith_chart_tex(filename):
    header = r"""
\begin{tikzpicture}
    % Define Okabe-Ito Colorblind Palette
    \definecolor{cbOrange}{RGB}{230, 159, 0}
    \definecolor{cbSky}{RGB}{86, 180, 233}
    \definecolor{cbGreen}{RGB}{0, 158, 115}
    \definecolor{cbYellow}{RGB}{240, 228, 66}
    \definecolor{cbBlue}{RGB}{0, 114, 178}
    \definecolor{cbVermillion}{RGB}{213, 94, 0}
    \definecolor{cbPurple}{RGB}{204, 121, 167}
    \definecolor{cbBlack}{RGB}{0, 0, 0}

    \begin{smithchart}[
        width=\linewidth,
        % Global font size for the axis labels
        %tick label style={font=\tiny},
        % Ensure the grid isn't too cluttered
        grid=both,
        major grid style={lightgray},
        legend style={
            at={(0.98,0.02)},
            anchor=south east,
            draw=black!20,
            fill=white,
            fill opacity=0.8,
            text opacity=1,
            font=\footnotesize,
            cells={anchor=west}
        }
    ]
    \end{smithchart}
\end{tikzpicture}
"""
    with open(filename, "w") as f:
        f.write(header)


def append_point_to_tex(filename, real, imag, color="cbBlack", legend=""):
    """Inserts a point and its legend entry."""
    new_plot = f"        \\addplot[{color}, mark=*, only marks] coordinates {{({real}, {imag})}};\n"
    if legend:
        new_plot += f"        \\addlegendentry{{{legend}}}\n"

    _insert_before_closing(filename, new_plot)

def append_circle_to_tex(filename_tex, filename_dat, color="cbBlue", style="solid", legend=""):
    """
    Inserts a circle with a specific color and line style.
    Example styles: 'dashed', 'dotted', 'dashdotted', 'thick, dashed'
    """
    # Note: we removed the 'node' part to keep the chart clean
    new_plot = f"        \\addplot [{color}, {style}, is smithchart cs] file {{{filename_dat}}};\n"
    if legend:
        new_plot += f"        \\addlegendentry{{{legend}}}\n"

    _insert_before_closing(filename_tex, new_plot)

def _insert_before_closing(filename, content):
    """Internal helper to insert text before the end of the smithchart environment."""
    if not os.path.exists(filename):
        create_smith_chart_tex(filename)
        
    with open(filename, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if r"\end{smithchart}" in line:
            lines.insert(i, content)
            break
            
    with open(filename, "w") as f:
        f.writelines(lines)

# def create_smith_chart_tex(filename="chart.tex"):
#     # Using a raw string (r"") to handle LaTeX backslashes correctly
#     latex_content = r"""
#     \begin{tikzpicture}
#         \centering
#         \begin{smithchart}[
#             width=14cm,
#             ]

#         \end{smithchart}
#     \end{tikzpicture}
#     """

#     try:
#         with open(filename, "w") as f:
#             f.write(latex_content)
#         print(f"Successfully created {filename}")
#     except Exception as e:
#         print(f"An error occurred: {e}")

# import os

# def append_point_to_tex(filename, real, imag, color="blue", legend=""):
#     """
#     Finds the \end{smithchart} in the file and inserts a new point above it.
#     """
#     if not os.path.exists(filename):
#         print(f"File {filename} not found. Initializing it first...")
#         create_smith_chart_tex(filename)

#     # Define the new line to insert
#     # Using f-string for variables and double {{ }} for LaTeX braces
#     new_plot_line = f"        \\addplot[{color}, mark=*] coordinates {{({real}, {imag})}} node[anchor=south west] {{{legend}}};\n"

#     with open(filename, "r") as f:
#         lines = f.readlines()

#     # Search for the closing tag to insert before it
#     insertion_index = -1
#     for i, line in enumerate(lines):
#         if r"\end{smithchart}" in line:
#             insertion_index = i
#             break

#     if insertion_index != -1:
#         # Insert the new line into the list of lines
#         lines.insert(insertion_index, new_plot_line)
        
#         # Write the lines back to the file
#         with open(filename, "w") as f:
#             f.writelines(lines)
#         print(f"Appended point ({real}, {imag}) to {filename}")
#     else:
#         print("Error: Could not find '\\end{smithchart}' in the file.")

# def append_circle_to_tex(filename_tex, filename_dat, color="blue", legend=""):
#     """
#     Finds the \end{smithchart} in the file and inserts the circle specified in the .dat file.
#     """
#     if not os.path.exists(filename_tex):
#         print(f"File {filename_tex} not found. Initializing it first...")
#         create_smith_chart_tex(filename_tex)

#     # Define the new line to insert
#     # Using f-string for variables and double {{ }} for LaTeX braces
#     new_plot_line = f"        \\addplot [{color},is smithchart cs,] file {{{filename_dat}}} node[below left] {{{legend}}};\n"

#     with open(filename_tex, "r") as f:
#         lines = f.readlines()

#     # Search for the closing tag to insert before it
#     insertion_index = -1
#     for i, line in enumerate(lines):
#         if r"\end{smithchart}" in line:
#             insertion_index = i
#             break

#     if insertion_index != -1:
#         # Insert the new line into the list of lines
#         lines.insert(insertion_index, new_plot_line)
        
#         # Write the lines back to the file
#         with open(filename_tex, "w") as f:
#             f.writelines(lines)
#         print(f"Appended circle from {filename_dat} to {filename_tex}")
#     else:
#         print("Error: Could not find '\\end{smithchart}' in the file.")

def generate_noise_figure_latex_table(F, CF, RF):
    """
    Generates a LaTeX table string from Noise Figure, 
    Center (polar tuples), and Radii arrays.
    """
    latex_str = (
        "\\begin{table}[H]\n"
        "\\centering\n"
        "\\begin{tabular}{|c|c|c|}\n"
        "\\hline\n"
        " $F$ (dB) & Center ($|\\Gamma_c| \\angle \\theta$) & Radius ($R_F$) \\\\\n"
        "\\hline\n"
    )

    for i in range(len(F)):
        # Unpack the polar tuple (magnitude, angle)
        mag, angle = CF[i]
        
        # Add a row to the table
        # Format: F | mag ∠ angle | radius
        row = (f" {F[i]:.2f} & {mag:.3f} $\\angle$ {angle:.1f}$^\\circ$ "
               f"& {RF[i]:.3f} \\\\\n")
        latex_str += row

    latex_str += (
        "\\hline\n"
        "\\end{tabular}\n"
        "\\caption{Noise Figure Circle Centers and Radii}\n"
        "\\label{tab:noise_circles}\n"
        "\\end{table}"
    )
    
    return latex_str

def generate_available_gain_latex_table(Ga, Ca, Ra):
    """
    Generates a LaTeX table string from available gain, 
    Center (polar tuples), and Radii arrays.
    """
    latex_str = (
        "\\begin{table}[H]\n"
        "\\centering\n"
        "\\begin{tabular}{|c|c|c|}\n"
        "\\hline\n"
        " $G_a$ (dB) & Center ($|\\Gamma_a| \\angle \\theta$) & Radius ($R_a$) \\\\\n"
        "\\hline\n"
    )

    for i in range(len(Ga)):
        # Unpack the polar tuple (magnitude, angle)
        mag, angle = Ca[i]
        
        # Add a row to the table
        # Format: F | mag ∠ angle | radius
        row = (f" {Ga[i]:.2f} & {mag:.3f} $\\angle$ {angle:.1f}$^\\circ$ "
               f"& {Ra[i]:.3f} \\\\\n")
        latex_str += row

    latex_str += (
        "\\hline\n"
        "\\end{tabular}\n"
        "\\caption{Available gain circles centers and radii}\n"
        "\\label{tab:gain_circles}\n"
        "\\end{table}"
    )
    
    return latex_str



def to_latex(name, value, fmt=None):
    """
    Convert a Python variable to a LaTeX \\newcommand.

    name : str   -> command name (without backslash)
    value        -> numeric or string
    fmt  : str   -> optional format spec, e.g. ".3f"
    """
    if isinstance(value, float) and fmt is not None:
        val_str = format(value, fmt)
    else:
        val_str = str(value)
    return f"\\newcommand{{\\{name}}}{{{val_str}}}"


def generate_filter_component_latex_table(data_string, precision=4):
    """
    Parses component values and converts them to SI prefixed values (e.g., 1.5 nH).
    """
    # Define SI prefixes
    prefixes = {
        -15: 'f',  # femto
        -12: 'p',  # pico
        -9: 'n',   # nano
        -6: r'\mu ', # micro (latex symbol)
        -3: 'm',   # milli
        0: '',     # units
        3: 'k',    # kilo
        6: 'M',    # mega
        9: 'G'     # giga
    }

    # Define Units based on first letter
    unit_map = {
        'L': 'H',
        'C': 'F',
        'R': r'$\Omega$',
        'F': 'Hz'
    }

    lines = data_string.strip().split('\n')
    latex_code = [
        # "\\begin{table}[H]",
        # "\\centering",
        "\\begin{tabular}{|l|l|}",
        "\\hline"
    ]

    for line in lines:
        if '=' not in line:
            continue
            
        name, val_str = line.split('=')
        name = name.strip()
        val = float(val_str.strip())

        # 1. Identify the base unit
        first_letter = name[0].upper()
        base_unit = unit_map.get(first_letter, "")

        # 2. Calculate the engineering exponent (multiples of 3)
        if val == 0:
            exp_3 = 0
            scaled_val = 0
        else:
            exponent = math.floor(math.log10(abs(val)))
            exp_3 = int(math.floor(exponent / 3) * 3)
            scaled_val = val / (10**exp_3)

        # 3. Get prefix symbol
        prefix_symbol = prefixes.get(exp_3, f"e{exp_3}")
        
        # 4. Format the row
        # Example: L1 & 1.567 nH \\ \hline
        formatted_row = f"{name} & {scaled_val:.{precision}f} {prefix_symbol}{base_unit} \\\\ \\hline"
        latex_code.append(formatted_row)

    latex_code.append("\\end{tabular}")
    # latex_code.append(f"\\caption{{{table_caption}}}")
    # latex_code.append("\\end{table}")

    return "\n".join(latex_code)

def append_string_to_tex(filename, content):
    """
    Checks if a .tex file exists, creates it if not, 
    and appends the provided string.
    """
    # 1. Ensure the filename ends with .tex
    if not filename.endswith('.tex'):
        filename += '.tex'

    # 2. Check if file exists to print a helpful message
    file_exists = os.path.exists(filename)
    
    try:
        # 3. Open in "append" mode ('a')
        # This creates the file if it doesn't exist
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
            f.write("\n") # Ensure the file ends with a newline
            
        if file_exists:
            print(f"Successfully appended content to {filename}")
        else:
            print(f"Created {filename} and wrote content.")
            
    except Exception as e:
        print(f"An error occurred: {e}")
