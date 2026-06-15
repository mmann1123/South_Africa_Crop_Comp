---
name: brief-literature-review
description: Write a general, publication-style literature review (writeup/literature/general_literature_review.md) synthesizing the per-paper briefer .md files into manuscript-ready prose. Use when asked to write a literature review, related-work section, background section, or a general academic survey of the crop-classification / remote-sensing literature. (For the PROJECT-CENTRIC findings synthesis instead, use brief-synthesize-crop-findings.)
---

Produce a general, field-facing literature review at `writeup/literature/general_literature_review.md` by synthesizing the per-paper briefer `.md` files in `writeup/literature/`. This SKILL.md is the canonical, self-contained source — follow the structure and rules below. The review synthesizes from the briefers, not the PDFs; if briefers are missing or thin on the evaluation-protocol line, fix those first with [brief-paper](../brief-paper/SKILL.md).

## Pick the right synthesis skill

Two synthesis products exist; choose deliberately:

- **This skill (`brief-literature-review`)** → `general_literature_review.md`. Academic, narrative, thematically organized. Written for readers of the manuscript or a proposal; our study appears only briefly, as positioning at the end. Reads like a related-work section.
- **[brief-synthesize-crop-findings](../brief-synthesize-crop-findings/SKILL.md)** → `0_findings_summary.md`. Project-centric, prescriptive, action-list. Organized around our crop-classification study's modeling choices and concrete TODOs. Internal decision support.

If the user wants prose for the paper → this skill. If they want "what should we do about our models" → the other one.

## Required structure

1. **Introduction and scope** — crop classification from satellite image time series, framed toward the Global South and dryland farming.
2. **Challenges to crop separability** — spectral similarity among crop types (e.g. cereals), phenological timing and irregular planting calendars, cloud cover and missing observations, smallholder field size and heterogeneity.
3. **Modeling approaches** — classical machine learning (Random Forest, SVM, gradient-boosted trees) on engineered/time-series features; deep learning (1D-CNN, LSTM/BiLSTM, temporal attention such as L-TAE, TempCNN, TabNet); patch/spatial models (2D/3D CNN, transformers); hybrid pipelines; optical-only vs SAR fusion.
4. **Representation and features** — handcrafted indices vs automated time-series feature extraction vs end-to-end learned representations; pixel vs field vs patch granularity.
5. **Evaluation practices and their pitfalls** — the review's strongest contribution. Audit how generalization is measured across the corpus: prevalence of pooled/random or field-wise k-fold within a single scene versus genuine spatially-disjoint holdouts; spatial-autocorrelation and FID leakage; class-imbalance artifacts; the in-region-vs-spatial-transfer gap.
6. **Knowledge gaps and future directions** — spatial/temporal transfer, domain adaptation, SAR-optical fusion, interpretability.
7. **Positioning (brief)** — where our Western Cape study sits (in-region vs spatial-holdout comparison; inductive-bias and transfer findings), 1–2 paragraphs only.
8. **References** — alphabetical, each linking to its briefer.

## Authoring rules

1. **Narrative prose, not bullet lists.** Paragraphs with topic sentences. Bullets only for genuinely enumerable items.
2. **Every claim cited** to a briefer via inline link; group co-supporting citations.
3. **Synthesize, don't enumerate.** Each paragraph advances a theme drawing on several papers — never a paper-by-paper walkthrough.
4. **Surface disagreements explicitly** (e.g. deep learning vs tree-ensemble superiority claims, optical-only vs SAR necessity) and explain them — usually a difference of scale or evaluation protocol.
5. **Discount by protocol, in prose.** Section 5 carries this; no tier tables.
6. **Stay field-facing.** Our study appears only in section 7.
7. **Academic register. No emojis. Markdown.** 1500–3000 words typical.

## Refresh vs full rebuild

- **Full rebuild**: read all briefers in parallel, write the review fresh.
- **Refresh** (new briefers added): read the new ones and weave them into the relevant thematic sections; do not append a "new papers" section.

## Reference

- Per-paper briefers (input): [writeup/literature/](../../../writeup/literature/)
- Project context for section 7: [CLAUDE.md](../../../CLAUDE.md) and [writeup/sn-article.tex](../../../writeup/sn-article.tex)
- Sibling skills: [brief-paper](../brief-paper/SKILL.md), [brief-synthesize-crop-findings](../brief-synthesize-crop-findings/SKILL.md), [briefer-bibliography](../briefer-bibliography/SKILL.md)
