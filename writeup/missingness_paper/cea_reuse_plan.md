# From the CEA manuscript to the missingness-shift paper: what to remove, update, and keep

Written 2026-11-09 against `../CEA_submission/cea-article.tex` and `cea-supplement.tex` (tag `cea-revision1-final`), the corrected results in `experiments/holdout_fill_check/results/table1_filled.csv`, and the follow-up experiments in `experiments/missingness/` and `experiments/sparse_gate_backbones/`. Line numbers are those of the CEA files. "A" is the article, "S" the supplement.

## 1. The numbers the new text rests on

Every number below is already computed; nothing here needs a GPU unless marked.

Holdout macro-F1 with September filled from training-tile medians, no retraining (`table1_filled.csv`; published value in brackets):

| Model | Level | In-region | OOR filled | OOR published | Gap (filled) |
|---|---|---|---|---|---|
| CNN-BiLSTM | pixel | 0.72 | 0.656 (0.513) | | −0.06 |
| TempCNN | pixel | 0.77 | 0.655 (0.574) | | −0.12 |
| L-TAE | pixel | 0.78 | 0.652 (0.539) | | −0.13 |
| L-TAE-S | pixel | 0.77 | 0.643 (0.599) | | −0.13 |
| Transformer | pixel | 0.78 | 0.641 (0.577) | | −0.14 |
| TimesNet | pixel | 0.76 | 0.633 (0.565) | | −0.13 |
| TabNet (raw) | pixel | 0.73 | 0.616 (0.603) | | −0.11 |
| TempCNN | field | 0.67 | 0.614 (0.535) | | −0.06 |
| XGBoost | field | 0.70 | 0.612 (0.565) | | −0.09 |
| LightGBM | field | 0.68 | 0.606 (0.559) | | −0.07 |
| MiniRocket | pixel | 0.72 | 0.605 (0.500) | | −0.11 |
| Logistic regression | pixel | 0.61 | 0.605 (0.563) | | −0.005 |
| L-TAE | field | 0.67 | 0.600 (0.594) | | −0.07 |
| L-TAE-S | field | 0.64 | 0.599 (0.601) | | −0.04 |
| Random Forest | pixel | 0.65 | 0.597 (0.548) | | −0.05 |
| LightGBM | pixel | 0.62 | 0.594 (0.563) | | −0.03 |
| CNN-BiLSTM | field | 0.64 | 0.591 (0.506) | | −0.05 |
| MiniRocket | field | 0.67 | 0.585 (0.482) | | −0.08 |
| TimesNet | field | 0.70 | 0.585 (0.516) | | −0.11 |
| Voting | pixel | 0.69 | 0.571 (0.557) | | −0.12 |
| XGBoost | pixel | 0.64 | 0.558 (0.544) | | −0.08 |
| Stacking | field | 0.64 | 0.555 (0.533) | | −0.09 |
| SMOTE Stacked | field | 0.69 | 0.543 (0.489) | | −0.15 |
| 3D CNN | patch | 0.70 | 0.592 (0.503) | | −0.11 |
| TabNet (raw, field-avg) | field | 0.62 | 0.489 (0.433) | | −0.13 |
| Multi-Channel 2D CNN | patch | 0.63 | 0.589 (0.446) | | −0.04 |
| TabNet (xr_fresh) | field | 0.51 | 0.405 (0.370) | | −0.11 |

Rank correlations (Spearman, 25 re-run models): in-region vs published OOR 0.36; in-region vs filled OOR 0.74; published vs filled OOR 0.56. The published all-field ranking correlates 0.86 with the ranking on the 1,063 September-missing fields and 0.33 with the ranking on the 1,352 September-observed fields (`experiments/missingness/results/subset_bootstrap.csv`). On September-observed fields the published predictions already rank CNN-BiLSTM first (0.659) and pixel TabNet 14th (0.617).

Other results the new paper uses: fill-value sweep (in-range fills interchangeable, median per-model spread 0.014, Spearman 0.89 to 0.97; zero fill applied to everyone costs raw models −0.075 and xr_fresh models −0.096 on average); September gap map and logit (the gap covers 76 percent of fields in the 8 to 16 km zone against 26 percent elsewhere; the distance effect of about −0.31 log-odds per 10 km vanishes once September share enters); month-dropout 2 x 2 (under zero fill dropout restores every backbone, +0.085 TempCNN, +0.162 CNN-BiLSTM, +0.117 L-TAE; the gate adds nothing on top; dense L-TAE + dropout 0.664 is the best single model); step deletion (equals the median fill within ±0.008 for L-TAE, L-TAE-S, Transformer); month-by-month injection (zero-filling any month breaks every non-dropout model, in-region as much as on the holdout; median fill and deletion agree that only October carries real information worth 0.04 to 0.16); flip rate (zero fill flips 37 percent of TempCNN and 64 percent of CNN-BiLSTM pixel predictions, 9 to 15 percent with dropout); learning curves to 5 percent (every model gains with data; raw-sequence models and logistic regression indistinguishable at 5 to 15 percent; XGBoost and TabNet trail logistic regression by 0.035 to 0.065 and 0.062 to 0.097; deep models separate from about 25 percent; no low-data gain from the gate); tree + deep ensemble (0.661 mixed vs 0.660 deep-only under the fill, n.s.).

The patch CNNs were corrected on 2026-11-09 (training-patch median fill; 3D CNN 0.592, 2D CNN 0.589, both mid-table). Figures and results that are *not* yet corrected and still rest on zero-filled predictions: the cereal merge (S-F, scores published predictions), the ensemble-size and loss/sampler ablations (S-D iv and v, re-inference without the fill switch at the time), the L-TAE-S hyperparameter grid (S-G), the MiniRocket gate experiment (S-I), the per-class figure, the error maps (S-E), and the raw-vs-xr_fresh XGBoost table (raw XGBoost read NaN September natively, so it is probably unaffected, but confirm `merged_dl_test.parquet` holds NaN rather than 0).

## 2. Article, section by section

### Front matter

Title (A:23), abstract (A:41), keywords (A:51), graphical abstract, and highlights are all replaced. Nothing in the abstract survives except the data description (South African Sentinel-2, optical only, 27 models, adjacent holdout tile, no target-region data). Keywords drop "feature selection" and add "missing data", "missingness shift", "imputation" or "cloud gaps".

### Introduction

| Passage | Fate | Reason |
|---|---|---|
| A:57, crop-classification context | Keep, trim | Still true. "Frequent cloud cover" moves from a list item to the paper's subject. |
| A:59, how generalization is measured, domain-generalization scoping, UDA citations | Keep most, add the missing half | The in-region vs out-of-region point stands, and the no-target-imagery scoping is still our setting. Add: transfer evaluations rarely check whether the target's observation pattern differs from training, and almost none report a per-domain missingness statistic (audit: 7 of about 33). |
| A:61, Western Cape test bed and model list | Keep | Rewrite the last clause: we evaluate twice, in-region and out-of-region, and the out-of-region tile misses one month for 41.5 percent of pixels. |
| A:63, central question | Replace | New question: when a model trained on complete monthly series is applied to a neighbouring tile where one month is missing for two fifths of pixels, how much of the apparent transfer loss is the missing month, how does the answer depend on how each pipeline fills the gap, and which remedies work? |
| A:65, five contributions | Replace all five | (1) inverted rankings: the inversion was the fill, not the region; (2) sparse feature selection transfers: not supported; (3) L-TAE-S beats attention capacity: not supported on consistent inputs (0.643 vs L-TAE 0.652); (4) aggregation as a lever: aggregation lowers filled OOR F1 for every architecture run at both levels; (5) data efficiency: the flat deep curve was an artifact. |
| A:68, Related Work | Rewrite second half | Keep the classical-vs-deep and engineered-vs-learned framing in one paragraph. Drop "whether richer representations generalize better ... is a central question here". Add a paragraph on missing observations in SITS models: masking and consistent interpolation in reference codebases (WorldCereal, SITS-BERT, U-TAE, EuroCropsML), observation dropout as training augmentation (Pham 2024 IJAEOG and RSE, Mask-PSTIN, Garnot's U-TAE occlusion), the train-complete/test-incomplete protocol (Wang, Belgiu 2025), and the term missingness shift (Zhou, Balakrishnan, Lipton 2023). State what is new: a cross-region, same-season case in which a single inconsistent fill reordered a multi-family benchmark, with the fill value separated from the information loss. |

Citations that must not carry over in their CEA role: `belgiu2016` for RF surviving transfer (its §6.1 says the opposite), `obadic2022` for attention dates being region-contingent (that paper varies the class set, not the region), and the "independent corroboration" set in A:268 (`pankajakshan2025`, `rustowicz2019`, `jin2019`, `mai2025`) as evidence that trees transfer. `mai2025` is the Hexi Corridor paper, rank 3 in our audit, which should not be cited approvingly for anything about transfer.

### Data

Keep A:74 to A:79 nearly whole. Add one paragraph and one table. The paragraph states the missing-data pattern: the training tiles miss only May, which every model treats as absent; the holdout misses September for 41.5 percent of pixels (B2), 40.2 percent of fields entirely and 44.0 percent in part, in a sharp-edged north-south strip consistent with a scene or cloud-mask boundary (`experiments/missingness/results/missing_data_by_month.csv`, `figures/sept_gap_error_map.pdf`). Use these figures, not the 40.6 percent or 39.3 percent quoted in earlier notes. The table is the per-month missing share for both regions (supplement if space is short). The sentence "the holdout season runs later" (A:79) stays; it is a real shift that remains after the correction.

### Methodology

| Passage | Fate |
|---|---|
| A:82, three families, field-wise splits, aggregation rule | Keep. |
| A:85, classical ML and xr_fresh | Keep, two fixes. Do not list autocorrelation among the features: the 18 `autocorr_lag_k` columns duplicate `absolute_sum_of_changes`, so 156 of the 174 features are distinct (say so). Add that the xr_fresh statistics are NaN-aware, so these models computed features on the nine observed months and never received a fill. |
| A:88, deep architectures | Keep the descriptions. Add the published fill rule: raw-sequence models received a reflectance of 0 for the missing month before the training-fitted scaler (CNN-BiLSTM: unscaled), and pixel TabNet received the holdout's own column medians. Move the TabNet sparsemax description down to one sentence. |
| A:90 to A:108, L-TAE-S | Keep as a method, shorten, reposition. It is no longer the proposed model; it is one of the remedies tested (a sparse channel gate). Keep the equations, move Fig. 2 to the supplement, delete "to test whether a tree-like, sparse inductive bias can be grafted onto a temporal-attention encoder" and the TabNet-mirroring sentences. The sentence on the inert λ penalty stays (it is a correction the reviewers asked for). |
| New subsection, fill operators and the consistent-fill re-analysis | One table per model (operator, statistic source, before or after scaling, retrained or re-scored, reproduces published predictions), from `experiments/missingness/results/fill_sheet.md`; main text or supplement. State the re-analysis: every model re-scored on the holdout with September filled from training-tile medians, no retraining, no holdout statistic. State the reproduction checks (every deep model reproduces its published predictions field for field; pixel TabNet's published file predates its saved checkpoints, 221 of 2,415 fields differ, 0.603 vs 0.609; classical features rebuilt and compared). |
| New subsection, experiments | Fill-value sweep (five fills); September-observed and September-missing field subsets; month-by-month injection; step deletion; the 2 x 2 month dropout x gate on three backbones; flip rate and integrated gradients; learning curves 5 to 100 percent; tree + deep vote. One paragraph each. |
| A:112, metrics | Keep. |
| A:115, uncertainty | Keep. Re-check the per-seed SD range (0.015 to 0.037) on the filled holdout; it will shrink for CNN-BiLSTM. |

### Results

**4.1 (A:121 to A:176).** The heading "In-Region Validation Overstates Performance and Misranks Models" and both paragraphs go. The published ranking is still shown, because the paper is about how it arose, but as the "before" column. Table 1 becomes: model, level, in-region F1, OOR published, OOR filled, gap on filled, 95 percent CI, κ, wF1. The two paired-bootstrap claims (TabNet over CNN-BiLSTM +0.09; over L-TAE +0.06) reverse sign and need recomputing on the filled predictions (CNN-BiLSTM over TabNet is about +0.04; `out_of_sample/paired_bootstrap_gaps.py` on `holdout_fill_check/predictions/`, a few minutes). Replace the text with three results in order: (a) the artifact: the south-central "error zone" is the September strip, September-observed fields inside it score like fields anywhere, and the distance effect vanishes once September share enters the logit; (b) the subsets: on September-observed fields the published predictions already rank the dense pixel networks first, and the published ranking tracks the September-missing subset (0.86) not the observed one (0.33); (c) the corrected ranking: dense pixel temporal models 0.63 to 0.66 lead trees, logistic regression, TabNet and MiniRocket at 0.59 to 0.62; in-region vs OOR rank correlation rises from 0.36 to 0.74; every model still loses under transfer, by 0.005 (logistic regression) to 0.14 (Transformer).

**4.2 Pixel vs field (A:178 to A:211).** Delete the narrative. On filled inputs aggregation lowers OOR F1 for all seven architectures run at both levels (TempCNN 0.655 to 0.614, L-TAE 0.652 to 0.600, L-TAE-S 0.643 to 0.599, CNN-BiLSTM 0.656 to 0.591, TimesNet 0.633 to 0.585, MiniRocket 0.605 to 0.585, TabNet 0.616 to 0.489). Keep Table 2 in the supplement with filled values, and one sentence in the main text: the published "aggregation helps L-TAE" came from averaging zero-filled pixels with observed ones inside a field, which pulled the field mean back toward the training range; with a sensible fill, averaging only removes information. (Mark the mechanism as the likely explanation, not measured.)

**4.3 Per-class (A:213 to A:221).** Regenerate `per_class_f1.pdf` from the filled predictions and change the four models (a dense pixel model, L-TAE, field XGBoost, pixel TabNet, or whichever set the text discusses). Delete every "gate improves every class" sentence. Keep the structural finding, with new numbers: lucerne/medics and canola map well, the three cereals are confused with each other in every model, and cereal errors account for most of the loss under transfer. The spatial-clustering sentence at the end becomes part of 4.1 (a).

**4.4 Data efficiency (A:223 to A:232).** Replace. The published flat L-TAE curve and the "inductive bias, not training volume" conclusion were the zero fill (the sweep was scored on the zero-filled holdout). Use `figures/learning_curves.pdf`: every model gains with data; at 5 to 15 percent of fields the raw-sequence models and logistic regression are indistinguishable (0.50 to 0.59), the two feature-selecting tabular models are the least data-efficient, and the deep models pull away from about 25 percent. State that draws are unpaired across models at 5 to 15 percent (different splits), so the null result there is weak; a paired rerun is pending a decision on how prominent this result will be.

**4.5 Raw vs xr_fresh (A:234 to A:261).** Keep, with a check: confirm that raw XGBoost saw NaN (handled natively) rather than 0 for September; if so the table stands and the conclusion (no accuracy gain from xr_fresh; its value is cost and interpretability) is unchanged. The TabNet half (0.43 vs 0.37) becomes 0.489 vs 0.405 on filled inputs, same direction. Drop "the best-transferring classical learners are exactly those fed xr_fresh", which was the motivation and no longer describes the ranking.

### Discussion

**5.1 Why tree-like inductive bias transfers (A:265 to A:277).** Delete the section and Fig. 5 (`inductive_bias_gap.pdf`). Nothing in it survives: the Transformer vs L-TAE-S comparison (0.641 vs 0.643 on filled inputs), the gate's 0.06 gain over L-TAE (now −0.009), the TimesNet and MiniRocket readings (MiniRocket now equals logistic regression at 0.605), the hyperparameter-sensitivity claim (grid scored on zero fill), the training-time parity claim (the cost facts are true, the parity is not). Replace with a section on what the artifact was and why it was invisible: a constant written into a month that training always observed, before a training-fitted scaler, confined to the target region, so in-region validation could not detect it, and the models with the largest "transfer gap" were those that received the out-of-range value. Then the fill-value sweep: any in-range fill, deletion, or interpolation gives the same ranking; the mixed treatment across pipelines is what produced the published one. Then the generality point from the injection experiment: zero-filling any month that training always observed breaks every non-dropout model, in-region as much as out, so this is not a September or transfer phenomenon.

**5.2 Two robustness levers (A:279 to A:280).** Delete. Replace with "Remedies": inference-time deletion (masking the month out of attention) equals the best fill with no retraining; training-time month dropout removes the damage under any fill and costs nothing on clean input, and dense L-TAE with dropout (0.664) is the best single model; the sparse gate is a partial substitute that dropout dominates (interaction −0.088 CNN-BiLSTM, −0.061 L-TAE); mixing trees and deep models in a vote hedges against the mismatch but is weaker than dropout. Place this against Pham 2024's Random Observation Selection and WorldCereal's simulated gaps as the standard remedy; our result is the cross-region version with the fill-value contrast and the gate comparison added.

**5.3 Automated features, patch models, persistent difficulty (A:283).** Keep the patch-CNN caveats and the cereal-difficulty paragraph; rewrite the opening (xr_fresh models are 0.04 below the dense networks, not "nearly as good as the best"; their value is cost). The patch CNNs are now corrected (0.59 each): they no longer "transfer poorly", they sit with the field-level models, below the pixel temporal networks; keep the resolution and receptive-field caveats as reasons they do not exceed pixel models. The cereal merge sentence needs re-scoring on filled predictions (`experiments/cereal_merge/cereal_merge.py` on `holdout_fill_check/predictions/`, minutes); "the sparse models remain at the top" goes.

**5.4 Scope and practical relevance (A:286).** Rewrite the three practical points. (1) Before attributing a transfer loss to the region, tabulate per-month observation coverage in both domains and score the target on observed and missing subsets separately. (2) Never fill a gap with a value outside the training distribution; mask it, interpolate, or use a training statistic, and do it the same way in every pipeline being compared. (3) If gaps are expected at deployment, train with month dropout; it costs nothing when the input is complete. Keep the per-crop guidance with filled numbers, and the cost point.

**5.5 Limitations (A:289 to A:291).** Keep the single-season, adjacent-tile, optical-only limits. Add: one dataset and one naturally missing month, with generality shown only by injection on the same data (a second dataset is the obvious next step); the pixel TabNet checkpoint mismatch; the xr_fresh duplicate features; the unpaired low-data comparisons.

**Conclusion (A:294).** Rewrite entirely. The CEA conclusion's recommendations (evaluate out-of-region; prefer feature-selecting models; test aggregation) become: evaluate out-of-region, but first check that the target's observation pattern matches training; report it; fill consistently or not at all; train for gaps.

### Back matter

CRediT, ethics, competing interests, acknowledgements carry over. Data availability adds the new script directories. The generative-AI declaration from the checklist is still owed.

## 3. Supplement, section by section

| CEA section | Fate |
|---|---|
| S-A dataset, features, model definitions (S:35 to S:255) | Keep. Fix Table S1 (autocorrelation duplicates). Add the per-month missing-data table and the fill sheet (one row per model). Move Fig. 2 (L-TAE-S architecture) here. |
| S-B metrics (S:256) | Keep. |
| S-C training cost (S:278) | Keep the table. Rewrite the lead: the sentence that the models that transfer best are the cheapest is no longer true; the dense pixel networks that lead cost 24 to 61 min per seed against seconds for the trees. |
| S-D ablations (S:346 to S:438) | (i) aggregation: rewrite with filled values. (ii) sparse selection vs capacity: delete. (iii) resampling: 0.555 vs 0.543 on filled inputs, direction unchanged, smaller. (iv) ensemble size and (v) loss/sampler: both were re-inference on the zero-filled holdout; rerun with `HOLDOUT_FILL=train_median` (a few GPU hours) or drop. The CNN-BiLSTM "swings 0.41 to 0.56 across objectives" is probably mostly the zero fill, so it cannot be kept as is. |
| S-E spatial errors (S:439) | Replace by the September gap map (`sept_gap_error_map.pdf`) and the zone-by-status table and logit from `experiments/missingness/README.md`. The reading "local conditions within the tile, dense networks most affected" was the artifact; say so. |
| S-F merged cereals (S:452) | Re-score on filled predictions, keep the table, delete "sparse models remain at the top". |
| S-G L-TAE-S hyperparameter sensitivity (S:507) | Drop, or keep one paragraph as a robustness-to-zero-fill result with that caveat stated. Every number was scored on zero fill. |
| S-H phenology (S:547) | Keep unchanged. |
| S-I recent baselines and MiniRocket gate (S:562) | Keep the implementation paragraph. Rewrite the comparison (TimesNet 0.633 pixel, MiniRocket 0.605, equal to logistic regression). The gate-on-MiniRocket table was scored under zero fill; either re-score the saved predictions under the training-median fill or present it as robustness to the out-of-range fill, next to the 2 x 2. |
| New | Fill-value sweep table (`fill_sweep/results/f1_by_fill.csv`); month-injection figure; step-deletion table; 2 x 2 month-dropout table with contrasts; flip-rate and integrated-gradients figure; learning-curve table with ensemble points; tree + deep vote table; codebase nodata audit summary (17 codebases, none reproduces the failure; say this plainly, it limits the claim). |

## 4. Framing

### The claim, sized to the evidence

The firm claim is narrow and the evidence for it is complete: in a 27-model out-of-region benchmark, one month missing for two fifths of target pixels, filled differently by different pipelines, produced an in-region to out-of-region ranking inversion that disappeared when every model received the same in-range fill. The models that appeared to overfit the training region were those that received a value outside the training distribution. This is a missingness shift (Zhou et al. 2023) confined to the target, which is why in-region validation could not catch it.

What must not be claimed: that this failure is common in public code (the 17-codebase audit found none that reproduces it), that it explains published transfer failures (the paper-level audit and four re-runs found gaps real but cheap in other benchmarks), or that we are first to show missingness passing for transfer loss (Pham 2024 showed it across years). The defensible novelty statement is the one in `reports/Missing data robustness novelty check.md`: first cross-region, same-season case in which an inference-time fill inconsistency reordered a multi-family SITS benchmark, with the fill value separated from the information loss (step deletion, injection) and the sparse gate measured against the standard remedy.

### Two framings, and a recommendation

**A re-analysis paper.** Lead with the before and after: the published ranking, the gap map, the corrected ranking, then the mechanism and the remedies. The benchmark is context. This is honest about how the finding arose and makes the error-detection recipe (observed vs missing subsets; per-domain coverage table; fill consistently) the contribution. Risk: it reads as an erratum unless the mechanism and remedy sections carry weight, which they do (injection, deletion, dropout 2 x 2, flip rates).

**A benchmark paper with a missingness section.** Lead with the corrected 27-model comparison (dense temporal models lead; rankings agree across protocols; learning curves), and report the missingness result as the main methodological finding. Risk: the corrected benchmark confirms BreizhCrops and WorldCereal, so the headline is not new, and the missingness result gets buried.

Recommendation: the re-analysis framing, with the benchmark as the setting. The injection experiment is what lifts it above a case study, so give it a full results subsection and a main-text figure. Put the month-dropout 2 x 2 in the main text as the remedy result, and state plainly that dropout dominates the gate; that sentence is the one a referee who read the CEA version will look for.

### Working title

The current working title in `article.tex` ("Cloud gaps at prediction time can masquerade as transfer failure in crop classification") is supported for our case but carries the general connotation the audit could not confirm. Alternatives that name the mechanism and the scope: "A missing month, not model family, reordered a 27-model crop-classification benchmark under out-of-region transfer"; "Missingness shift at prediction time reverses a crop-classification model ranking: a re-analysis with remedies"; "Inconsistent gap filling, not transfer, separated deep and classical crop classifiers on an unlabeled tile".

### Tone

State the error once, early, in the paper's own voice (Introduction, last paragraph, and a sentence in Methods on the fill sheet), then write the rest as findings. Do not apologise in the results. The reproduction checks (every deep model reproduces its published predictions field for field) belong in Methods because they are what makes the before/after comparison trustworthy.

## 5. Work still needed before drafting

Cheap, from saved predictions (minutes): paired bootstraps between the top filled models and TabNet, XGBoost, logistic regression; per-class F1 figure on filled predictions; cereal merge on filled predictions; per-seed SD on filled holdout; confirm raw-XGBoost September handling.

Cheap, GPU (hours): ensemble-size and loss/sampler ablations under `HOLDOUT_FILL=train_median`, or drop them; MiniRocket gate re-score under the fill.

Decisions (taken 2026-11-09): the low-data null claim is kept, and the paired 5 to 15 percent sweep is running (`experiments/missingness/learning_curves/paired/`); the patch CNNs are being corrected with a training-patch median fill (`experiments/holdout_fill_check/run_deep_field.py --models mc2d cnn3d`); the target journal is Computers and Electronics in Agriculture.
