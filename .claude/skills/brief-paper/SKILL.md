---
name: brief-paper
description: Create a per-paper briefer markdown file for a crop-classification / remote-sensing PDF in writeup/literature/. Use when asked to brief a paper, summarize a PDF, add a paper to the literature folder, or create a literature briefer.
---

Generate one markdown briefer per PDF in `writeup/literature/`. This SKILL.md is the canonical, self-contained source — follow the template and rules below exactly.

This corpus supports the manuscript `writeup/sn-article.tex` — a systematic comparison of classical machine-learning, deep-learning, and patch-based crop classifiers on multi-temporal Sentinel-2 imagery in the Western Cape of South Africa. The paper's central finding is that conventional in-region (field-wise k-fold) validation systematically misranks models, while a spatially disjoint holdout tile reorders them: sparse, feature-selecting and tree-like models (gradient-boosted trees, TabNet) transfer best, whereas dense temporal/patch deep nets overfit the training region. Briefers should be written through that lens.

## When to invoke

The user names one or more PDFs (paths or filenames matching `writeup/literature/*.pdf`) and asks for a briefer / summary / synopsis. If asked to brief many papers at once (N ≥ 5), parallelize via sub-agents, one agent per group of ~5 PDFs, each agent following this same template.

## What you produce

For each PDF, one markdown file at `writeup/literature/<same-base-name>.md` — same base filename as the PDF, `.pdf` replaced by `.md`, all other characters preserved (spaces, punctuation, em-dashes). Each briefer uses this exact section template:

- `# {Paper Title}`
- `**Citation:**` (authors, year, venue, DOI)
- `## Objectives`
- `## Methods` (including the load-bearing **Evaluation protocol** line — see rule 1)
- `## Key Findings`
- `## Relevance to Our Crop-Classification Study`
- `## Evaluation Caveats`

## Authoring rules

1. **Evaluation protocol is the load-bearing section.** State precisely how the paper measured generalization, and where it sits on the spectrum that this manuscript is built around: pooled/random k-fold over all pixels → field-wise (FID) k-fold within one scene → a spatially disjoint holdout (a different tile, region, or agroecological zone) → cross-year or cross-sensor transfer. Most crop-classification papers report the first two and call them "test-set" or "out-of-sample" accuracy; flag in-region-k-fold-disguised-as-OOS explicitly, because it is exactly the conflation this paper exists to correct.
2. **Flag spatial leakage.** The cardinal sin is pixels from the same field (or adjacent, spatially autocorrelated pixels) appearing in both train and test, which inflates accuracy without measuring transfer. Note whether splitting was done field-wise (FID-disjoint) and whether train/test are spatially separated at all.
3. **Flag class-imbalance artifacts.** Crop datasets are dominated by a majority class (here lucerne/medics). Note overall-accuracy figures that mask poor minority-crop recall, and whether the paper reports a balanced metric (macro-F1, per-class F1, Cohen's Kappa) rather than only accuracy or weighted F1.
4. **Cite the protocol, not the headline F1.** e.g. "Their 0.92 overall accuracy comes from random k-fold over pooled pixels within a single scene and is not comparable to our spatially-disjoint-holdout macro-F1."
5. **Note what the paper does NOT measure.** Spatial transfer, cross-year robustness, per-class minority performance, and computational cost are frequently absent — silences are findings.
6. **Note feature and sensor design** relevant to us: optical-only vs SAR fusion, band/index choice, raw temporal sequences vs engineered time-series features (e.g. xr_fresh-style statistics), pixel vs field vs patch granularity.
7. **For reviews/theory papers**, skip the protocol question and focus Relevance on methodological/conceptual implications (e.g. inductive bias, domain transfer, interpretability).
8. **Markdown only. No emojis. Identifiers and band names in backticks.** Roughly 250–400 lines per briefer.
9. **Unparseable PDF**: write a stub with `## Status: Could not extract` rather than skipping.
10. **DOI extraction and verification (required).** Extract the DOI from the PDF (first page, header/footer, or abstract block — formats `10.xxxx/...`, `https://doi.org/...`, `DOI:`). Verify it by calling `WebFetch` on `https://api.crossref.org/works/<doi>` and confirming the returned `title` matches (case-insensitive substring is fine). If Crossref 404s or the title diverges, tag the DOI `unverified` and add a one-line note. If no DOI is in the PDF, try one Crossref title search (`https://api.crossref.org/works?query.title=<title>&rows=3`); if a top hit matches, use it tagged `(from Crossref title match)`. Otherwise write `DOI: not found in PDF`. Never fabricate a DOI. The first place to look for an existing DOI is `writeup/literature/references.bib` (the machine-built bibliography), followed by [writeup/literature/README.md](../../../writeup/literature/README.md) and the manuscript's `thebibliography` block — but treat any DOI found there as a lead, not ground truth: still verify it against Crossref before trusting it. This project's bibliography has already contained fabricated, retracted, and wrong-metadata entries, so a DOI being present in the `.bib` or README is not evidence that it is correct.

## After writing the briefer(s)

If a new briefer materially changes the set of directly comparable papers (a new true-spatial-holdout comparator, or a feature/architecture finding that bears on the paper's argument), follow up with [brief-synthesize-crop-findings](../brief-synthesize-crop-findings/SKILL.md) to refresh `writeup/literature/0_findings_summary.md`. Otherwise the briefer alone suffices.

## Reference

- Project context to inject: [CLAUDE.md](../../../CLAUDE.md) and the manuscript [writeup/sn-article.tex](../../../writeup/sn-article.tex).
- Cite keys / DOIs for the corpus: [writeup/literature/README.md](../../../writeup/literature/README.md).
- Especially relevant comparators in the corpus: `pankajakshan2026` ("Deep Architectures Fail to Generalize: A Lightweight Alternative for Agricultural Domain Transfer") — a true cross-domain transfer study, directly on-point; `rustowicz2019` (smallholder crop segmentation in Ghana and South Sudan) — Global-South, data-scarce; `cropformer2023` (generalized multi-scenario crop classification incl. transfer learning).
- Sibling skills: [brief-literature-review](../brief-literature-review/SKILL.md), [brief-synthesize-crop-findings](../brief-synthesize-crop-findings/SKILL.md), [briefer-bibliography](../briefer-bibliography/SKILL.md).
