# Missingness-shift manuscript

Successor to the withdrawn CEA submission (frozen at git tags `cea-submitted` and
`cea-revision1-final`; source in `../CEA_submission/`). Text is being written fresh around the
September missingness-shift finding.

- `article.tex`: manuscript (elsarticle, author-year); section outline in comments.
- `missingness.bib`: starts as a copy of `../CEA_submission/cea.bib`.
- `figures`: symlink to the shared `../figures/`.
- Build: `./build.sh`.

Supporting analysis: `experiments/holdout_fill_check/`, `experiments/sparse_gate_backbones/`,
`experiments/missingness/`. Literature: `../literature/0_findings_summary.md` and
`reports/Missing data robustness novelty check.md`.
