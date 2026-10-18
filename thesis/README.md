# Polish thesis sources

`thesis-pl.tex` is the main document. Build from this directory with pdfLaTeX and
Biber as in the root [README](../README.md). Polish text, tables, bibliography and
the included `titlepage-pl.pdf` are preserved. Commented-out chapter and appendix
inputs remain commented out.

`thesis-pl.pdf` is the supplied main PDF. Its duplicate build-directory PDF was
removed after verifying identical SHA-256 hashes. The duplicate standalone title
page PDF was also removed; `titlepage-pl.pdf` is the canonical copy.
`title_page/` retains its standalone sources and images.

Figures identical to notebook outputs are shared from `../results/` using explicit
relative LaTeX paths. Unique EDA figures remain in `figures/`.
Keep `results/` alongside `thesis/` and `presentation/` when building documents.
For experiment reruns, use `STEAM_RESULTS_DIR` as described in the root README to
protect the published figure files. The supplied PDFs were not rebuilt.

`template/` preserves original template notices and historical skeleton
instructions. They describe the earlier skeleton, not the populated thesis.
No repository-wide license is inferred from the template. LaTeX auxiliaries were
removed; deliberate assets and PDFs remain.
