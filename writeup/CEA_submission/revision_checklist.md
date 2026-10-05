# COMPAG-D-26-07034 — Major Revision Checklist

**Manuscript:** Sparse Feature Selection, Not Deep Learning Capacity, Drives ~~Spatial~~ Out-of-Region Transfer in Crop Classification
**Journal:** Computers and Electronics in Agriculture (AE: Fernando Auat Cheein)
**Resubmission deadline:** 2026-11-01

Effort tiers: **Simple** = text edits only; **Medium** = new figures/analysis from existing models and data; **Complex** = new experiments or retraining.

**Documentation convention.** Each completed item records file:line (in the *revised* files, so later edits may shift numbers; the quoted snippets are the stable reference) and **before → after** wording, for use in the response letter. `A` = `cea-article.tex`, `S` = `cea-supplement.tex`, `H` = `highlights.tex`.

## Pre-flight findings

- [x] **Wrong abstract in working copy.** `cea-article.tex` lines 40–58 contain a California wildfire abstract, introduced in commit `11930a8 clean up text`. The correct crop-classification abstract is in commit `107c3c5 CEA submitted`. Restore before any other edits.
- [x] **R1-4 is confirmed in code.** `experiments/field_reduction/models_arch.py:288` computed `sparsity_loss = masks.mean()`. Because sparsemax outputs sum to 1, this is the constant 1/E and contributes no gradient; the λ term was inert. Sparsity actually comes from sparsemax plus the γ relaxation prior. Resolved under R1-4 below.

## Simple (text only)

- [x] **Wrong abstract** — restore from `107c3c5`.
- [x] **R2-Q1 — Title overstates "Spatial Transfer."** Term changed to "out-of-region transfer" (abbrev. ST → OOR) throughout, defined and scoped explicitly.
  - **Global rename** (A, S, H): "spatial transfer"/"spatial-transfer" → "out-of-region transfer"; "ST" → "OOR". Covers keywords (A:51), Table 2 caption and header "Spatial transfer (ST)" → "Out-of-region (OOR)" (A:129, A:132), Table pix/field caption (A:178), §4.4 heading "Data Efficiency Under Out-of-Region Transfer" (A:215) and caption (A:222), Table repr rows "Spatial transfer" → "Out-of-region" (A:244–245), Fig. bias caption, S ablation captions (S:365, S:396).
  - **Title** (A:23, S:26): "…Drives Spatial Transfer in Crop Classification" → "…Drives Out-of-Region Transfer in Crop Classification".
  - **Abstract** (A:41): "under spatial transfer to a disjoint holdout tile" → "under out-of-region transfer, applying them without any target-region labels or adaptation to an adjacent, non-overlapping holdout tile"; "must be evaluated across regions rather than within one" → "should be evaluated outside the region they were trained on"; "must generalize to unlabeled farmland" → "must be applied to nearby unlabeled farmland".
  - **Introduction** (A:61): "on a spatially disjoint holdout tile" → "on an adjacent, non-overlapping holdout tile". (A:63) central question "when evaluated on a spatially disjoint region" → "when applied, without target-region labels, to an adjacent region outside the training tiles"; "best ST macro-F1" → "best out-of-region macro-F1". (A:66) "under genuine spatial transfer" → "under out-of-region transfer".
  - **Data / definition** (A:75): "a relatively *local* spatial shift" → "a relatively *local* shift"; "``spatial-transfer'' (ST) or ``holdout'' denotes predictions on the disjoint tile." → "``out-of-region'' (OOR) or ``holdout'' denotes predictions on the adjacent, non-overlapping tile, made without any target-region labels or adaptation. We use ``out-of-region'' rather than the broader ``spatial transfer'' because the holdout tests a local shift, not transfer across agroecological zones or years."
  - **Metrics** (A:117): "spatial transfer (ST) to the disjoint holdout tile" → "out-of-region transfer (OOR) to the adjacent, non-overlapping holdout tile".
  - **Limitations** (A:278), added: "Our conclusions should therefore be read as applying to out-of-region transfer between neighboring tiles, not to cross-ecozone or cross-year transfer."
  - **Conclusion** (A:281): "applied to a spatially disjoint holdout" → "applied to an adjacent, non-overlapping holdout tile"; "evaluate on a spatially disjoint holdout" → "evaluate on an out-of-region holdout"; "when mapping unseen terrain" → "when mapping nearby unlabeled areas".
  - **Highlights** (H:9): "Sparse, axis-aligned feature selection, not deep capacity, drives spatial transfer." → "Sparse feature selection, not deep capacity, drives out-of-region transfer." (shortened to ≤85 chars). (H:12) "spatially disjoint holdout tests" → "out-of-region holdout tests". (H:13) "Crop models for new farmland must be evaluated across regions, not within one." → "Crop models should be tested outside their training region, not only within it."
  - **Incidental fix** (A:59): typo "target-domain data: labels: unlabeled imagery" → "labels, unlabeled imagery".
  - Unchanged on purpose: "spatially disjoint" where it describes tile geometry or prior studies (A:59, A:66, A:72).
  - **Still to do:** regenerate 5 figures with "spatial transfer"/"ST" baked in: `study_area_map`, `inductive_bias_gap`, `per_class_f1`, `field_reduction_multidraw`, `graphical_abstract`.
- [x] **R1-1 — Inconsistent model-category wording.** Adopted a single two-level taxonomy (classical ML, 9 models; deep learning, 14 = deep temporal 12 + deep patch-based 2, counted from Table 2) and dropped "hybrid" as a category.
  - **Methods** (A:80): "We compare three paradigms: feature-based classical machine learning, representation-learning deep networks, and patch-based spatial models." → "We compare two model families: classical machine learning on engineered xr_fresh features (9 models) and deep learning (14), the latter comprising deep temporal models on per-pixel or per-field time series (12) and deep patch-based models on image patches (2). The inductive-bias grouping in the Discussion cuts across these families."
  - **Abstract** (A:41): "twenty-three classical and deep models" → "twenty-three classical machine-learning and deep-learning models".
  - **Introduction** (A:61): "We evaluate CNN-BiLSTM hybrids, TabNet ensembles, lightweight temporal attention (L-TAE) and full Transformer-encoder networks, temporal-convolution (TempCNN) networks, and patch-based 2D/3D CNNs, benchmarked against XGBoost, LightGBM, and RF." → "We evaluate deep temporal models (CNN-BiLSTM, TabNet, L-TAE, a Transformer encoder, and TempCNN) and deep patch-based 2D/3D CNNs against classical XGBoost, LightGBM, and RF."
  - **Contributions** (A:63): "(twenty-three classical, deep, and patch-based models)" → "(twenty-three classical, deep temporal, and deep patch-based models)"; central question "across machine-learning, deep-learning, and hybrid architectures" → "across classical machine-learning and deep-learning models".
  - **Related Work** (A:66): "Hybrid designs combine…" → "Other designs combine…".
  - **Methods, deep models** (A:86): "a CNN-BiLSTM hybrid trained with" → "a CNN-BiLSTM trained with"; "field-level hybrid-voting rule" → "field-level voting rule"; "Patch-based models (a multi-channel 2D CNN and a 3D CNN)" → "Deep patch-based models (…)".
  - **Supplement training-cost table** (S:310, S:318, S:326): "Deep, field-level" / "Deep, pixel-level" / "Deep, patch-level" → "Deep temporal, field-level" / "Deep temporal, pixel-level" / "Deep patch-based".
- [x] **R1-4 — λ‖m‖₁ penalty is constant under sparsemax.** Reviewer is correct. Option 1 taken: text corrected and dead code removed; **no results change and no retraining needed.**
  - **Verification** (deep_field env, float64, untrained L-TAE-S): mask sums = 1.000 (±4×10⁻¹⁶); old penalty value = 0.0078125 = 1/128 exactly; max |gradient difference| with vs. without the term = 2.8×10⁻¹⁷ (machine precision) against max |gradient| = 0.17. So all reported L-TAE-S results were effectively trained with λ = 0. (Also: 98.6% of mask entries are exactly zero even untrained, so "most channels … are exactly zero" stands.)
  - **Main text, Methods §3.2.1** (A:99): "…discourages successive heads from re-selecting the same channels; a penalty $\lambda\sum_{h,t}\lVert\mathbf{m}_{h,t}\rVert_{1}$ ($\lambda=10^{-3}$) adds mild additional pressure." → "…discourages successive heads from re-selecting the same channels. These two mechanisms are the sole source of sparsity; no explicit sparsity penalty is used, because an $\ell_1$ penalty on a sparsemax mask is constant ($\lVert\mathbf{m}_{h,t}\rVert_1=1$) and has zero gradient."
  - **Supplement, TabNet paragraph** (S:206): TabNet was *not* affected (pytorch_tabnet uses a mask-entropy penalty, which varies with mask concentration; library default λ_sparse = 10⁻³ is used, as no script overrides it). Added for clarity: "Unlike an $\ell_1$ norm, which is fixed at 1 on the simplex, TabNet's mask-entropy penalty $\lambda_{\text{sparse}}\sum_i\sum_j -M_j[i]\log M_j[i]$ ($\lambda_{\text{sparse}}=10^{-3}$, library default) varies with mask concentration and so adds genuine sparsity pressure."
  - **Code** (`experiments/field_reduction/`): `models_arch.py` — `LTAESparse.forward` no longer computes `sparsity_loss = masks.mean()`; returns `aux = {"masks": masks}` with a docstring explaining why. `train_epoch_sparse` now adds `lambda_sparse * aux["sparsity_loss"]` only if the model supplies one (FastTabNet's entropy penalty still applies). Removed `LAMBDA_SPARSE` and the `lambda_sparse=` argument / metadata key from `train_ltae_sparse_field.py`, `train_ltae_sparse_pixel.py`, `resume_ltae_sparse_pixel.py`, `seed_sweep.py`. Existing checkpoints load unchanged (the term had no parameters).
  - **Response-letter point:** thank the reviewer; confirm the analysis; state the term was constant with zero gradient, so reported results already reflect the model without it; text and public code corrected.
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
- [ ] ~~**R1-4 (alternative) — Make λ effective.**~~ Not pursued (Option 1 chosen). Could still add an entropy-penalty setting to the R1-2 sensitivity sweep.
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
