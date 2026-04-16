If latex.py is used to make smitcharts for latex documents the following should be added to the preamble:

\usepackage{circuitikz}
\usepackage{pgfplots}
\pgfplotsset{width=7cm,compat=1.18}\usepgfplotslibrary{smithchart}

And then smitcharts can be loaded in latex as:

\begin{figure}[H]
    \centering
    \input{filename.tex}
    \caption{<Caption>}
    \label{<label>}
\end{figure}

Where "filename.tex" is where the python generated latex code is saved.