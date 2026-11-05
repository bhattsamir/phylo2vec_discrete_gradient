#!/usr/bin/env bash
set -euo pipefail

# Build the editable vector figure before the manuscript that includes it.
paper_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$paper_dir/figures"
latexmk -pdf -interaction=nonstopmode -halt-on-error exact_block_overview.tex
cd "$paper_dir"
latexmk -pdf -interaction=nonstopmode -halt-on-error phylo2vec_bme_paper.tex
