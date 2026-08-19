#!/usr/bin/env bash
# Build the CEA article, supplement, highlights, and cover letter.
# Author-year (Harvard) citations require pdflatex -> bibtex -> pdflatex x2.
set -e
cd "$(dirname "$0")"

build() {
  local base="$1"
  echo "==> Building $base"
  pdflatex -interaction=nonstopmode -halt-on-error "$base.tex" >/dev/null
  bibtex "$base" >/dev/null || true
  pdflatex -interaction=nonstopmode -halt-on-error "$base.tex" >/dev/null
  pdflatex -interaction=nonstopmode -halt-on-error "$base.tex" >/dev/null
}

build cea-article
build cea-supplement
pdflatex -interaction=nonstopmode -halt-on-error highlights.tex >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error cover-letter.tex >/dev/null

echo "Done. PDFs: cea-article.pdf, cea-supplement.pdf, highlights.pdf, cover-letter.pdf"
