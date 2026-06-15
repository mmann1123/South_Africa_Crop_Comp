---
name: briefer-bibliography
description: Build writeup/literature/references.bib from every per-paper briefer .md in writeup/literature/, fetching authoritative BibTeX from Crossref (and arXiv as fallback) and validating each entry. Use when asked to build the bibliography, generate references.bib, regenerate the .bib file, or refresh citations after new briefers have been added.
---

Build `writeup/literature/references.bib` from every per-paper briefer markdown file in `writeup/literature/`. The skill extracts the DOI from each briefer's `**Citation:**` line, fetches authoritative BibTeX from Crossref (falling back to arXiv for preprints), validates each returned entry, normalizes citation keys, and writes a per-briefer status log.

## When to invoke

- "Build references.bib" / "regenerate the bibliography" / "make a .bib file from the briefers"
- After [brief-paper](../brief-paper/SKILL.md) adds new briefer(s) and the bibliography needs to catch up
- After DOIs are added or corrected in existing briefers

## What it produces

- `writeup/literature/references.bib` — one BibTeX entry per briefer that has a resolvable DOI or arXiv ID, alphabetized by citation key
- `writeup/literature/references.log` — per-briefer status (`OK`, `NO_DOI`, `FETCH_FAIL`, `PARSE_FAIL`, `INCOMPLETE`, `TITLE_MISMATCH`, `DOI_DIVERGE`) so failures are auditable
- A new line `**BibTeX key:** \`<citekey>\`` injected into each briefer immediately below its `**Citation:**` line. This is the linkage used to refer to a paper without re-deriving the key from the .bib file. The injection is idempotent — re-running updates an existing key line rather than duplicating it.

Note: the manuscript `writeup/sn-article.tex` currently uses a hand-maintained `thebibliography` block with its own cite keys (e.g. `ieee2017`, `cropformer2023`, `tabnet_orig`); those keys are listed in [writeup/literature/README.md](../../../writeup/literature/README.md). `references.bib` is the machine-built companion. If you intend the two to share keys, reconcile the generated keys with the manuscript's keys (or keep `references.bib` as a separate verified source).

### Manual entries (datasets, code, grey literature)

Not everything cited has a briefer. Datasets and repositories (Radiant Earth MLHub, the Spot-the-Crop / Western Cape competition dataset, Sentinel-2 / Google Earth Engine, the project code repo) and grey literature do not go through the per-paper briefer pipeline but still need BibTeX entries for the manuscript. To keep everything in one `.bib` file without losing manual entries on regeneration:

- The script writes auto-generated (briefer-derived) entries at the top of `references.bib`.
- Below them sits a sentinel line: `% ===== MANUAL ENTRIES BELOW — preserved across build_bib.py regenerations =====`.
- Anything appended after the sentinel is **preserved** when `build_bib.py` runs again. Auto-generated entries above the sentinel are replaced wholesale.

To add a manual entry: open `writeup/literature/references.bib`, scroll to the sentinel, and append a `@misc{...}`, `@techreport{...}`, `@dataset{...}`, or other BibTeX entry below it. The next `build_bib.py` run will leave it alone.

### How to cite a paper in prose

When writing in `0_findings_summary.md`, `general_literature_review.md`, or any LaTeX export, use the citation key from the briefer:

- Markdown / informal: `[Wang et al. 2023](<cropformer2023_Wang.md>)` paired with the BibTeX key `wang2023cropformer` when emitting LaTeX
- LaTeX: `\cite{wang2023cropformer}` (the `.bib` entry already exists in `references.bib`)

If a briefer does not yet have a `**BibTeX key:**` line, run this skill — that means the bibliography hasn't been regenerated since the briefer was added or its DOI was corrected.

## How to invoke

```bash
python /home/mmann1123/Documents/github/South_Africa_Crop_Comp/.claude/skills/briefer-bibliography/build_bib.py
```

Optional flags:

- `--lit-dir PATH` — override the literature directory (default: `writeup/literature/` relative to repo root)
- `--out PATH` — override output `.bib` path
- `--log PATH` — override output `.log` path
- `--sleep SEC` — pause between Crossref requests (default 0.2 s; raise if rate-limited)
- `--strict` — exit non-zero if any briefer fails to resolve (useful for CI)

No `pip install` step required. The script uses Python stdlib only (`urllib`, `re`, `pathlib`) so it runs in any Python ≥ 3.8.

## What "rock-solid" means here

The script enforces six validation layers per briefer:

1. **Exclusion of non-briefer files.** Index pages (`0_findings_summary.md`, `general_literature_review.md`), prompt files (`*_prompt.md`), `README.md`/`MEMORY.md`, and generated outputs are skipped automatically.
2. **DOI extraction is strict.** The DOI is pulled from the `**Citation:**` line via the Crossref regex (`10.\d{4,9}/...`). If the briefer has no DOI (or only `DOI: not found`), the script falls back to an `arXiv:NNNN.NNNNN` ID. If neither is present, the briefer is logged `NO_DOI` and skipped — never fabricated.
3. **Crossref BibTeX endpoint is canonical.** Fetched from `https://api.crossref.org/works/<doi>/transform/application/x-bibtex` with a polite-pool `User-Agent`. If Crossref returns a non-BibTeX payload, the script falls back to `https://doi.org/<doi>` with content negotiation. Both must agree on a `@...{...}` opener.
4. **Required-fields check.** Each entry must parse as BibTeX AND contain `author`, `title`, and a year. Missing any triggers `INCOMPLETE` and the entry is excluded.
5. **DOI roundtrip.** The DOI inside the returned BibTeX must match the DOI sent (case-insensitive). Divergence is logged `DOI_DIVERGE` but not fatal.
6. **Title overlap.** Briefer H1 title vs. BibTeX `title=` must share ≥ 40% of meaningful (non-stopword) tokens. The stopword list excludes the most ubiquitous domain words (`crop`, `crops`, `classification`, `remote`, `sensing`) so the comparison rests on distinguishing terms. Below threshold triggers `TITLE_MISMATCH` so a human can audit a possible bad DOI.

Successful entries are re-keyed to a deterministic `<lastauthor><year><titleword>` form (e.g. `wang2023cropformer`), collision-suffixed if needed, and written sorted alphabetically.

## Reading the log

One briefer per line, with a single status tag at the start:

| Tag | Meaning | Fix |
|---|---|---|
| `OK` | Entry written. Includes the citation key and the Crossref URL used. | none |
| `NO_DOI` | Citation line has neither a `10.x/x` DOI nor an `arXiv:` ID. | Add DOI to briefer (run [brief-paper](../brief-paper/SKILL.md) DOI step) or accept omission |
| `FETCH_FAIL` | All resolution endpoints returned an error or non-BibTeX response. | Check internet, check that DOI is real, retry |
| `PARSE_FAIL` | Returned text was BibTeX-shaped but did not parse. | Inspect raw response |
| `INCOMPLETE` | BibTeX missing one of `author`, `title`, year. | Likely Crossref-side metadata gap; add manual entry |
| `DOI_DIVERGE` | Crossref returned a slightly different DOI string than requested. | Usually safe; verify the entry visually |
| `TITLE_MISMATCH` | Title from Crossref doesn't overlap with briefer's H1 title. | Strong signal the DOI in the briefer is wrong — verify and correct |

## Integration with sibling skills

- [brief-paper](../brief-paper/SKILL.md) — produces the briefer files this skill consumes; also enforces DOI verification at brief-time so most DOIs are already validated by the time this runs.
- [brief-literature-review](../brief-literature-review/SKILL.md) and [brief-synthesize-crop-findings](../brief-synthesize-crop-findings/SKILL.md) — consumers of `references.bib` when their outputs are exported to LaTeX.

## After a successful run

If the log shows any non-`OK` rows, surface them to the user explicitly. `TITLE_MISMATCH` rows in particular indicate a likely briefer-side DOI error that should be corrected before the `.bib` file is trusted downstream.
