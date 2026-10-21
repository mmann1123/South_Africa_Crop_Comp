#!/usr/bin/env bash
# Build the missingness-shift manuscript (elsarticle, author-year citations).
set -e
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode -halt-on-error article.tex >/dev/null
bibtex article >/dev/null || true
pdflatex -interaction=nonstopmode -halt-on-error article.tex >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error article.tex >/dev/null
echo "Done: article.pdf"
