# writeup/CLAUDE.md

LaTeX manuscript conventions. Loads when working under `writeup/`.

- LaTeX manuscripts and figures. Two variants: the active IEEE TGRS submission (`tgrs-article.tex` + online supplement `tgrs-supplement.tex`, S-numbered floats) and the original Springer Nature version (`sn-article.tex`).
  - **Table notes go *below* the table, never in the `\caption{}`.** The caption holds a short title only; put all explanatory notes (column/symbol definitions, †/‡ footnotes, "n=…", method caveats, "bold = best") in a `\footnotesize` block beneath the tabular: `\end{tabular}\par\vspace{3pt}\begin{minipage}{\columnwidth}\footnotesize <note>\end{minipage}\end{table}` (use `\textwidth` for `table*`). Applies to both manuscripts.
  - **Group closely related figures into one multi-panel float** (panels labelled (a), (b), … with one shared caption) instead of many small separate floats, to keep the supplement readable.
  - Main↔supplement cross-references are hardcoded (`\ref` does not cross files), so re-number S-figures/S-tables by hand and re-grep the main text after any supplement float change.
