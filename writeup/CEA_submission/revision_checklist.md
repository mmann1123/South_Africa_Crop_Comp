# COMPAG-D-26-07034 — Major Revision Checklist

**Manuscript:** Sparse Feature Selection, Not Deep Learning Capacity, Drives Spatial Transfer in Crop Classification
**Journal:** Computers and Electronics in Agriculture (AE: Fernando Auat Cheein)
**Resubmission deadline:** 2026-11-01

Effort tiers: **Simple** = text edits only; **Medium** = new figures/analysis from existing models and data; **Complex** = new experiments or retraining.

## Pre-flight findings

- [x] **Wrong abstract in working copy.** `cea-article.tex` lines 40–58 contain a California wildfire abstract, introduced in commit `11930a8 clean up text`. The correct crop-classification abstract is in commit `107c3c5 CEA submitted`. Restore before any other edits.
- [ ] **R1-4 is confirmed in code.** `experiments/field_reduction/models_arch.py:288` computes `sparsity_loss = masks.mean()`. Because sparsemax outputs sum to 1, this is the constant 1/E and contributes no gradient; the λ term is inert. Sparsity actually comes from sparsemax plus the γ relaxation prior.

## Simple (text only)

- [x] **Wrong abstract** — restore from `107c3c5`.
- [ ] **R2-Q1 — Title overstates "Spatial Transfer."** Retitle to reflect the no-target-label, adjacent-region setting (e.g. "…Drives Out-of-Region Transfer Without Target Labels in Crop Classification").
- [ ] **R1-1 — Inconsistent model-category wording.** Use "classical ML / deep temporal / deep patch-based" consistently in the abstract, Introduction (line 80), Table 2 caption, and Conclusion.
- [ ] **R1-4 — λ‖m‖₁ penalty is constant under sparsemax.** Correct the Methods text: state that λ is inert and sparsity arises from sparsemax and the γ prior. Check the supplement's TabNet description for the same issue. (See Medium for the retraining alternative.)
- [ ] **R1-5 — xr_fresh vs. raw tested only with XGBoost.** State this scope explicitly in §4.5, Discussion, and Conclusion; do not imply it holds for deep temporal models.
- [ ] **R1-6 — Patch-CNN failure attributed only to limited context and lack of sparsity.** Soften the causal claim; name patch size, receptive field, and spatio-temporal fusion strategy as alternative explanations.
- [ ] **R1-3 — +0.02 F1 gain is below the paper's own ~0.05 significance threshold.** Reframe "sparsity drives transfer" as "is associated with / is consistent with"; rest the argument on the family-level evidence (inductive-bias figure) rather than the L-TAE vs. L-TAE-S difference. Coordinate with the title change.
- [ ] **R1-8 / R2-Q7 — Adjacent, same-year, same-ecozone holdout.** Expand Limitations to state the domain of validity explicitly (transfer distance, year, agro-ecological zone, area size).
- [ ] **R2-Q5 — 0.60 macro-F1 is not operational.** Add a practical-use paragraph: model selection when no target labels exist, per-class reliability (canola/lucerne usable, cereals not), and low compute cost.
- [ ] **AE — Worldwide state-of-the-art review.** Broaden the literature search (2023–2026 cross-region / domain-generalization crop-mapping work from China, Europe, the Americas) and remove marginally related citations.
- [ ] **R1 / R2-Q9 — Language editing.** Full copy-edit pass (`copyedit-manuscript` skill).
- [ ] **AE — Detailed response letter.** Point-by-point, with line references to the revised manuscript. Do last.

## Medium (new figures/analysis, existing models)

- [ ] **R2-Q2 — Show crop phenology and whether time series alone separate the classes.** Add a figure of monthly mean ± SD EVI (plus B11/B12) per crop, training vs. holdout; discuss which pairs overlap (wheat / barley / small-grain grazing). Source: `data/merged_dl_train.parquet`, `data/merged_dl_test.parquet`.
- [ ] **R2-Q4 — Maps of classification results.** Multi-panel map of the holdout tile: ground truth, L-TAE-S, TabNet, XGBoost, CNN-BiLSTM, and a correct/incorrect panel. Add a short spatial-error analysis (clustering, tile-edge effects, field size). Source: `out_of_sample/predictions_*.csv` + test label GeoJSON.
- [ ] **R1-7 — No analysis of the hard cereal classes.** Discuss the cereal-triad confusion matrix; optionally test hierarchical classification or report F1 with a merged "cereal" class.
- [ ] **R1-4 (alternative) — Make λ effective.** Replace `masks.mean()` with an entropy penalty (as in TabNet), retrain 5 seeds of L-TAE-S, and report the result. The text fix alone is acceptable if time is short.
- [ ] **R2-Q7 (partial) — Effect of training-area size.** Reframe the existing field-reduction experiment (25/50/75%) as the area-size analysis and point to it in the response.

## Complex (new experiments)

- [ ] **R1-2 — Hyperparameter sensitivity of L-TAE-S.** Sweep γ ∈ {1.0, 1.3, 1.5, 2.0}, n_head ∈ {4, 8, 16}, and λ (once effective), 5 seeds each; report ST macro-F1 as a supplementary table or heatmap.
- [ ] **R2-Q3a — Recent crop time-series baselines.** Add 1–2 recent models (e.g. TimesNet, PatchTST, a Presto-style pretrained encoder, or a lightweight UTAE/TSViT variant) under the same protocol.
- [ ] **R1-5 (optional) — xr_fresh vs. raw for a temporal model.** Run L-TAE (or TabNet) on xr_fresh features vs. raw sequences. Optional; the scope edit already answers the comment.
- [ ] **R1-6 (optional) — Patch-size experiment.** Retrain the 2D/3D CNNs at 2–3 patch sizes (e.g. 50/100/200 m). Expensive; softening the text is the cheaper route.
- [ ] **R1-8 / R2-Q3b / R2-Q7 — Cross-year, cross-ecozone, or longer-distance transfer.** AI4FoodSecurity SA has only 2017 labels for these tiles, so cross-year requires another dataset (e.g. AI4FoodSecurity Germany/Brandenburg, EuroCrops, Sen4AgriNet). Cheaper middle option: rotate which of the three tiles is held out, giving three transfer pairs instead of one.

## Suggested order

1. Restore the abstract.
2. Complete the Simple text edits (they address roughly half the comments).
3. Produce the phenology figure and the prediction maps (both explicitly requested by R2).
4. Run the L-TAE-S hyperparameter sweep and the rotated-holdout experiment (best credibility per unit effort among the Complex items).
5. For cross-year, give a clear data-availability justification alongside the tile rotation.
6. Write the response letter.

## Submission deliverables

Upload via Editorial Manager → "Submissions Needing Revision" by **2026-11-01**. Suggested working folder: `revisions_1/` (currently empty). Items marked *(verify)* are standard Elsevier revision requirements; confirm against the current CEA Guide for Authors and the upload screen before submitting.

### Required

- [ ] **Response to reviewers** (`revisions_1/response-to-reviewers.pdf`). The AE asked for a *very detailed* letter: quote every comment (AE, R1-1…R1-8, R2-Q1…R2-Q7), then give the response, the change made, and page/line numbers in the revised manuscript. Note explicitly where a request was answered by scoping instead of a new experiment (e.g. cross-year transfer) and why.
- [ ] **Revised manuscript, marked-up** (`revisions_1/cea-article-marked.pdf`), with changes highlighted. Generate with `latexdiff` against `107c3c5:writeup/CEA_submission/cea-article.tex`. *(verify)*
- [ ] **Revised manuscript, clean** (`cea-article.pdf`), plus editable LaTeX source (`cea-article.tex`, `cea.bib` or `.bbl`, `build.sh` output) since Elsevier needs source files for production. *(verify)*
- [ ] **Revised supplementary material** (`cea-supplement.pdf`, plus a marked-up version if the supplement changed substantially). Re-check the hardcoded main ↔ supplement S-numbers after adding new S-figures/S-tables (sensitivity sweep, phenology, maps).
- [ ] **Highlights** (`highlights.tex` → PDF): 3–5 bullets, ≤85 characters each including spaces. Update them to match the new title and softened "associated with" claim.
- [ ] **Graphical abstract** (`figures/graphical_abstract.pdf/.png`): update if the title or headline claim changes. Check Elsevier's minimum size (531 × 1328 px, h × w). *(verify)*
- [ ] **Figure files**, uploaded separately if requested, including the new phenology and map figures, at ≥300 dpi (vector PDF preferred for line art).
- [ ] **Declaration of competing interest** (`declarationStatement.docx`): re-upload, regenerating it if authorship changes.
- [ ] **Title-page and metadata updates in Editorial Manager**: the new title must match the manuscript, and keywords should be revised if the framing changed.

### In-manuscript statements to re-check

Already present in `cea-article.tex` at lines 283–301.

- [ ] CRediT authorship contribution statement (update if new co-authors or new experiments).
- [ ] Declaration of competing interest.
- [ ] Data availability: confirm the repo link covers the new experiment scripts (sensitivity sweep, tile rotation, maps).
- [ ] Acknowledgements and funding.
- [ ] **Declaration of generative AI use in the writing process.** This is not currently in the manuscript. Elsevier requires it, placed before the references, if AI tools were used to prepare the text. *(verify)*

### Optional

- [ ] Cover letter for the revision (`cover-letter.tex`): a short summary of the major changes and the new title, pointing to the response letter.
- [ ] Research Elements companion submission (code/data), which the decision letter invites. It may carry an APC.
