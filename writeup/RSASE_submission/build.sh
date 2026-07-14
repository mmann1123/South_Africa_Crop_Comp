#!/usr/bin/env bash
# Build the RSASE article, supplement, and highlights.
# Author-year (Harvard) citations require the pdflatex -> bibtex -> pdflatex x2 sequence.
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

build rsase-article
build rsase-supplement
pdflatex -interaction=nonstopmode -halt-on-error highlights.tex >/dev/null

echo "Done. PDFs: rsase-article.pdf, rsase-supplement.pdf, highlights.pdf"
