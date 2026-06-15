---
name: brief-synthesize-crop-findings
description: Build or refresh writeup/literature/0_findings_summary.md — the PROJECT-CENTRIC synthesis organized around our crop-classification study's modeling choices, evaluation bar, and action items. Use when asked to refresh the findings summary, rebuild the literature synthesis, or update 0_findings_summary. (For a general academic literature review instead, use brief-literature-review.)
---

Build (or update) `writeup/literature/0_findings_summary.md` by reading every per-paper briefer `.md` in `writeup/literature/` and synthesizing across them. This SKILL.md is the canonical, self-contained source — follow the structure and rules below. The synthesis reads from the briefers, not the PDFs, because the briefers carry the evaluation-protocol audit the synthesis depends on. If briefers are missing or sparse on the evaluation-protocol line, fix those first with [brief-paper](../brief-paper/SKILL.md); don't synthesize from incomplete briefers.

## When to invoke

- User asks to "refresh the findings summary" / "rebuild the literature synthesis" / "update 0_findings_summary".
- A new briefer materially changes the set of directly comparable papers.
- Project criteria change (model menu, evaluation protocol, target metrics) — the boilerplate context paragraph and the action list need re-auditing.

## Organizing principle

**Evaluation protocol is the single most important lens.** Papers are not comparable unless they share a protocol. This study's whole argument is that in-region validation misranks models relative to a spatially disjoint holdout, so the synthesis keeps protocol front and centre:

- Section 1 sorts every paper into Tier 1 (true spatial holdout — different tile/region/zone, or cross-year/cross-sensor transfer) / Tier 2 (field-wise FID k-fold within a single scene — no spatial transfer) / Tier 3 (pooled or random k-fold — spatial-autocorrelation/FID leakage) / Tier 4 (reviews and theory).
- Section 2 names the 1–5 papers credibly directly comparable to our study (Western Cape Sentinel-2, field-level crop classification, ideally with a spatial holdout).
- Sections 3–4 extract feature-design and model-architecture findings, discounted by source tier.
- Section 5 conceptual/theoretical framing (inductive bias and transfer, interpretability, optical-vs-SAR).
- Section 6 project-specific caveats surfaced by the literature audit (e.g. our holdout is an adjacent tile, so our gaps are a conservative lower bound on cross-region transfer; EVI computed per-scene then composited).
- Section 7 prioritized action list (high lift / medium / lower), naming concrete files, features, or scripts.
- Section 8 bottom-line numerical targets the literature actually supports (macro-F1, Kappa under spatial transfer).

## Authoring rules

1. **Discount every reported number by protocol tier.** Never quote a number without naming the protocol that produced it.
2. **Group by finding, not by paper.** Multiple papers per point, multiple findings per paper.
3. **Link every paper reference** to its briefer file (URL-encoded filename). Add a citation-key block at the top mapping cite keys to briefers.
4. **Section 2 is the most important** — direct comparators define realistic targets.
5. **Section 7 must be actionable** — name files / features / scripts (e.g. `out_of_sample/`, `experiments/field_reduction/`, the gap table in `sn-article.tex`).
6. **Section 8 must take a stand** — state specific macro-F1 / Kappa targets the literature supports under genuine spatial transfer.
7. **Distinguish what the literature supports from what it cannot speak to** — silences (spatial transfer, minority-crop recall, cross-year robustness) are findings.
8. **Update, don't append.** Edit existing sections in place when refreshing.
9. **Markdown only. No emojis.**

## Refresh vs full rebuild

- **Full rebuild** (corpus changed substantially or first build): read all briefers in parallel, write the document fresh.
- **Refresh** (one or two new briefers): read just the new briefers, edit the relevant sections — tier assignment in section 1, possible promotion to section 2, feature/architecture points in sections 3–4, an action item in section 7.

## Reference

- Per-paper briefer skill: [brief-paper](../brief-paper/SKILL.md)
- Project context for the boilerplate paragraph: [CLAUDE.md](../../../CLAUDE.md) and [writeup/sn-article.tex](../../../writeup/sn-article.tex)
- Sibling skills: [brief-literature-review](../brief-literature-review/SKILL.md), [briefer-bibliography](../briefer-bibliography/SKILL.md)
