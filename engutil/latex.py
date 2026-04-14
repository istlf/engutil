import numpy as np
import os

def create_smith_chart_tex(filename="chart.tex"):
    # Using a raw string (r"") to handle LaTeX backslashes correctly
    latex_content = r"""
    \begin{tikzpicture}
        \centering
        \begin{smithchart}[
            width=14cm,
            ]

        \end{smithchart}
    \end{tikzpicture}
    """

    try:
        with open(filename, "w") as f:
            f.write(latex_content)
        print(f"Successfully created {filename}")
    except Exception as e:
        print(f"An error occurred: {e}")

import os

def append_point_to_tex(filename, real, imag, color="blue", legend=""):
    """
    Finds the \end{smithchart} in the file and inserts a new point above it.
    """
    if not os.path.exists(filename):
        print(f"File {filename} not found. Initializing it first...")
        create_smith_chart_tex(filename)

    # Define the new line to insert
    # Using f-string for variables and double {{ }} for LaTeX braces
    new_plot_line = f"        \\addplot[{color}, mark=*] coordinates {{({real}, {imag})}} node[anchor=south west] {{{legend}}};\n"

    with open(filename, "r") as f:
        lines = f.readlines()

    # Search for the closing tag to insert before it
    insertion_index = -1
    for i, line in enumerate(lines):
        if r"\end{smithchart}" in line:
            insertion_index = i
            break

    if insertion_index != -1:
        # Insert the new line into the list of lines
        lines.insert(insertion_index, new_plot_line)
        
        # Write the lines back to the file
        with open(filename, "w") as f:
            f.writelines(lines)
        print(f"Appended point ({real}, {imag}) to {filename}")
    else:
        print("Error: Could not find '\\end{smithchart}' in the file.")

def append_circle_to_tex(filename_tex, filename_dat, color="blue", legend=""):
    """
    Finds the \end{smithchart} in the file and inserts the circle specified in the .dat file.
    """
    if not os.path.exists(filename_tex):
        print(f"File {filename_tex} not found. Initializing it first...")
        create_smith_chart_tex(filename_tex)

    # Define the new line to insert
    # Using f-string for variables and double {{ }} for LaTeX braces
    new_plot_line = f"        \\addplot [{color},is smithchart cs,] file {{{filename_dat}}} node[below left] {{{legend}}};\n"

    with open(filename_tex, "r") as f:
        lines = f.readlines()

    # Search for the closing tag to insert before it
    insertion_index = -1
    for i, line in enumerate(lines):
        if r"\end{smithchart}" in line:
            insertion_index = i
            break

    if insertion_index != -1:
        # Insert the new line into the list of lines
        lines.insert(insertion_index, new_plot_line)
        
        # Write the lines back to the file
        with open(filename_tex, "w") as f:
            f.writelines(lines)
        print(f"Appended circle from {filename_dat} to {filename_tex}")
    else:
        print("Error: Could not find '\\end{smithchart}' in the file.")


