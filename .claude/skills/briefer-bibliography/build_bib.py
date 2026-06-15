#!/usr/bin/env python3
"""Build references.bib from per-paper briefer .md files in writeup/literature/.

Stdlib-only. Run:

    python build_bib.py

Process per briefer:
  1. Extract title (H1) and DOI / arXiv ID from the **Citation:** line.
  2. Fetch authoritative BibTeX from Crossref:
       https://api.crossref.org/works/<doi>/transform/application/x-bibtex
     Fall back to doi.org content negotiation for DOIs, or arXiv API for
     preprints.
  3. Validate: must parse, must contain author + title + year, DOI must
     roundtrip, title must overlap with briefer's title (>= 40% non-stopword
     tokens).
  4. Re-key to <lastauthor><year><titleword> and write to references.bib
     sorted alphabetically.
  5. Write per-briefer status to references.log.
"""

import argparse
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

# ----- Configuration ------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent  # .claude/skills/briefer-bibliography/ -> repo
LIT_DIR_DEFAULT = REPO_ROOT / "writeup" / "literature"

USER_AGENT = "crop-classification-bibliography/1.0 (mailto:mmann1123@gmail.com)"

MANUAL_SENTINEL = "% ===== MANUAL ENTRIES BELOW — preserved across build_bib.py regenerations ====="

EXCLUDE_PREFIXES = ("0_", "1_")
EXCLUDE_SUFFIXES = ("_prompt.md",)
EXCLUDE_NAMES = {"general_literature_review.md", "MEMORY.md", "README.md"}

DOI_RE = re.compile(
    r"\b(10\.\d{4,9}/[-._;()/:A-Z0-9]+)",
    re.IGNORECASE,
)
ARXIV_RE = re.compile(
    r"arXiv\s*[:.]?\s*(\d{4}\.\d{4,5})(?:v\d+)?",
    re.IGNORECASE,
)

STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "for", "and", "with", "using", "via",
    "from", "to", "by", "at", "into", "through", "based", "across", "over",
    "under", "is", "are", "be", "as",
    # Domain-specific filler often appearing in titles
    "crop", "crops", "classification", "remote", "sensing",
}

TITLE_OVERLAP_THRESHOLD = 0.40

# ----- Briefer parsing ----------------------------------------------------


def find_briefers(lit_dir: Path):
    out = []
    for path in sorted(lit_dir.glob("*.md")):
        name = path.name
        if name in EXCLUDE_NAMES:
            continue
        if any(name.startswith(p) for p in EXCLUDE_PREFIXES):
            continue
        if any(name.endswith(s) for s in EXCLUDE_SUFFIXES):
            continue
        out.append(path)
    return out


def parse_briefer(path: Path):
    """Return (title, doi_or_None, arxiv_id_or_None)."""
    title = None
    citation_line = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if title is None and line.startswith("# "):
            title = line[2:].strip()
        if citation_line is None and line.lstrip().startswith("**Citation:**"):
            citation_line = line
        if title and citation_line:
            break
    doi = None
    arxiv = None
    if citation_line:
        m = DOI_RE.search(citation_line)
        if m:
            doi = m.group(1).rstrip(").,;:]>")
        m = ARXIV_RE.search(citation_line)
        if m:
            arxiv = m.group(1)
    return title, doi, arxiv


# ----- HTTP helpers -------------------------------------------------------


def _http_get(url: str, accept: str, timeout: float = 25.0) -> str:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": accept},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def fetch_bibtex_from_doi(doi: str, timeout: float = 25.0):
    """Try Crossref then doi.org content negotiation. Return (bibtex, source)."""
    encoded = urllib.parse.quote(doi, safe="/")
    sources = [
        ("crossref", f"https://api.crossref.org/works/{encoded}/transform/application/x-bibtex"),
        ("doi.org", f"https://doi.org/{encoded}"),
    ]
    last_err = None
    for label, url in sources:
        try:
            data = _http_get(url, accept="application/x-bibtex", timeout=timeout).strip()
            if data.lstrip().startswith("@"):
                return data, f"{label}:{url}"
            last_err = f"{label}: non-bibtex response (first 100 chars: {data[:100]!r})"
        except urllib.error.HTTPError as e:
            last_err = f"{label}: HTTP {e.code}"
        except urllib.error.URLError as e:
            last_err = f"{label}: URL error {e.reason}"
        except TimeoutError as e:
            last_err = f"{label}: timeout {e}"
        time.sleep(0.5)
    raise RuntimeError(last_err or f"all sources failed for {doi}")


def fetch_bibtex_from_arxiv(arxiv_id: str, timeout: float = 25.0):
    """Build a @misc BibTeX entry from arXiv's Atom API."""
    url = f"http://export.arxiv.org/api/query?id_list={urllib.parse.quote(arxiv_id)}"
    xml_text = _http_get(url, accept="application/atom+xml", timeout=timeout)
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(xml_text)
    entry = root.find("a:entry", ns)
    if entry is None:
        raise RuntimeError(f"arxiv API returned no entry for {arxiv_id}")
    title = (entry.findtext("a:title", default="", namespaces=ns) or "").strip()
    title = re.sub(r"\s+", " ", title)
    published = (entry.findtext("a:published", default="", namespaces=ns) or "").strip()
    year = published[:4] if published else ""
    authors = []
    for a in entry.findall("a:author", ns):
        nm = a.findtext("a:name", default="", namespaces=ns).strip()
        if nm:
            authors.append(nm)
    if not title or not authors or not year:
        raise RuntimeError(f"arxiv entry for {arxiv_id} missing title/author/year")
    author_field = " and ".join(authors)
    placeholder_key = f"arxiv{arxiv_id.replace('.', '_')}"
    bibtex = (
        f"@misc{{{placeholder_key},\n"
        f"  title = {{{title}}},\n"
        f"  author = {{{author_field}}},\n"
        f"  year = {{{year}}},\n"
        f"  eprint = {{{arxiv_id}}},\n"
        f"  archivePrefix = {{arXiv}},\n"
        f"}}"
    )
    return bibtex, f"arxiv:{url}"


# ----- BibTeX parsing -----------------------------------------------------


def parse_bibtex(bibtex: str):
    """Lightweight BibTeX field extractor.

    Returns {"type": str, "key": str, "fields": {name: value}}.
    Supports brace-delimited and quote-delimited values; not a full grammar.
    """
    head = re.match(r"\s*@(\w+)\s*\{\s*([^,]+),", bibtex)
    if not head:
        raise ValueError("not a bibtex entry")
    entry_type = head.group(1).lower()
    key = head.group(2).strip()
    body = bibtex[head.end():]

    fields = {}
    i = 0
    n = len(body)
    while i < n:
        # Skip whitespace and commas
        while i < n and body[i] in ", \t\n\r":
            i += 1
        # End of entry
        if i < n and body[i] == "}":
            break
        # Read field name
        m = re.match(r"(\w+)\s*=\s*", body[i:])
        if not m:
            i += 1
            continue
        fname = m.group(1).lower()
        i += m.end()
        if i >= n:
            break
        delim = body[i]
        if delim == "{":
            # Brace-balanced extraction
            depth = 1
            i += 1
            start = i
            while i < n and depth > 0:
                if body[i] == "{":
                    depth += 1
                elif body[i] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                i += 1
            value = body[start:i]
            i += 1  # consume closing }
        elif delim == '"':
            i += 1
            start = i
            while i < n and body[i] != '"':
                i += 1
            value = body[start:i]
            i += 1
        else:
            # Bare number (e.g. year = 2024)
            start = i
            while i < n and body[i] not in ", \t\n\r}":
                i += 1
            value = body[start:i]
        fields[fname] = value.strip()
    return {"type": entry_type, "key": key, "fields": fields}


# ----- Citation key + title overlap --------------------------------------


def normalize_token(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def first_meaningful_word(title: str) -> str:
    for tok in re.split(r"[\s\-:]+", title):
        nt = normalize_token(tok)
        if not nt or nt in STOPWORDS or len(nt) < 3:
            continue
        return nt
    return ""


def make_citekey(parsed_fields: dict) -> str:
    author = parsed_fields.get("author", "")
    year = parsed_fields.get("year") or parsed_fields.get("date", "")
    title = parsed_fields.get("title", "")

    last = ""
    if author:
        first = author.split(" and ")[0].strip()
        if "," in first:
            last = first.split(",")[0].strip()
        else:
            # Take last whitespace-separated token, strip braces
            last = re.sub(r"[{}]", "", first).split()[-1] if first else ""
    last = normalize_token(last) or "anon"

    year_norm = ""
    if year:
        ym = re.search(r"\d{4}", year)
        if ym:
            year_norm = ym.group(0)

    word = first_meaningful_word(title) or "paper"
    return f"{last}{year_norm}{word}" if year_norm else f"{last}_{word}"


def title_token_overlap(briefer_title: str, bibtex_title: str) -> float:
    def tokens(s: str):
        return {
            normalize_token(t)
            for t in re.split(r"[\s\-:]+", s)
            if normalize_token(t) and normalize_token(t) not in STOPWORDS
        }
    a, b = tokens(briefer_title), tokens(bibtex_title)
    if not a:
        return 0.0
    return len(a & b) / len(a)


def rekey_bibtex(bibtex: str, new_key: str) -> str:
    return re.sub(
        r"^(\s*@\w+\s*\{)\s*[^,]+,",
        lambda m: f"{m.group(1)}{new_key},",
        bibtex,
        count=1,
    )


# Preferred field order for emitted entries. Anything not in this list goes
# after, alphabetized.
FIELD_ORDER = [
    "author", "title", "year", "month", "date",
    "journal", "booktitle", "publisher", "edition", "volume", "number", "pages",
    "doi", "url", "eprint", "archiveprefix", "issn", "isbn",
    "note", "abstract",
]


CITEKEY_LINE_RE = re.compile(r"^\*\*BibTeX key:\*\*\s+`([^`]+)`\s*$")


def inject_citekey_into_briefer(path: Path, citekey: str) -> str:
    """Insert/update `**BibTeX key:** `<citekey>`` immediately after the
    briefer's **Citation:** line. Returns one of: 'inserted', 'updated',
    'unchanged'. Idempotent.
    """
    lines = path.read_text(encoding="utf-8").splitlines(keepends=False)
    new_line = f"**BibTeX key:** `{citekey}`"

    citation_idx = None
    for i, line in enumerate(lines):
        if line.lstrip().startswith("**Citation:**"):
            citation_idx = i
            break
    if citation_idx is None:
        return "unchanged"

    # Existing key line immediately after Citation (possibly separated by a blank)
    insert_at = citation_idx + 1
    # Skip a single optional blank line so we can check the next non-blank
    probe = insert_at
    if probe < len(lines) and lines[probe].strip() == "":
        probe += 1
    if probe < len(lines):
        m = CITEKEY_LINE_RE.match(lines[probe].strip())
        if m:
            if m.group(1) == citekey:
                return "unchanged"
            lines[probe] = new_line
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return "updated"

    # No existing key line — insert one. Place it on the line directly after
    # Citation, preceded by a blank line for readability if needed.
    insert_block = [new_line]
    if insert_at < len(lines) and lines[insert_at].strip() != "":
        insert_block.append("")  # blank between key and following content
    if lines[citation_idx].strip() != "":
        # Already non-empty Citation line; ensure a blank wasn't expected
        pass
    lines[insert_at:insert_at] = insert_block
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return "inserted"


def format_bibtex(entry_type: str, key: str, fields: dict) -> str:
    """Re-emit a BibTeX entry with one field per line, aligned and ordered."""
    ordered_names = [f for f in FIELD_ORDER if f in fields]
    extras = sorted(f for f in fields if f not in FIELD_ORDER)
    names = ordered_names + extras

    lines = [f"@{entry_type}{{{key},"]
    for name in names:
        value = fields[name].strip()
        # Collapse internal whitespace runs (Crossref sometimes ships newlines
        # inside title fields).
        value = re.sub(r"\s+", " ", value)
        lines.append(f"    {name} = {{{value}}},")
    lines.append("}")
    return "\n".join(lines)


# ----- Main --------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lit-dir", type=Path, default=LIT_DIR_DEFAULT)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--log", type=Path, default=None)
    ap.add_argument("--sleep", type=float, default=0.2)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args(argv)

    lit_dir = args.lit_dir.resolve()
    out_path = args.out or (lit_dir / "references.bib")
    log_path = args.log or (lit_dir / "references.log")

    if not lit_dir.is_dir():
        print(f"ERROR: lit-dir not found: {lit_dir}", file=sys.stderr)
        return 2

    briefers = find_briefers(lit_dir)
    print(f"Found {len(briefers)} briefer files in {lit_dir}")

    entries = {}      # citekey -> bibtex string
    log_lines = []
    counts = {"OK": 0, "NO_DOI": 0, "FETCH_FAIL": 0, "PARSE_FAIL": 0,
              "INCOMPLETE": 0, "TITLE_MISMATCH": 0, "DOI_DIVERGE": 0}

    for briefer in briefers:
        rel = briefer.name
        title, doi, arxiv = parse_briefer(briefer)

        # Choose resolution path
        try:
            if doi:
                bibtex, source = fetch_bibtex_from_doi(doi)
            elif arxiv:
                bibtex, source = fetch_bibtex_from_arxiv(arxiv)
            else:
                counts["NO_DOI"] += 1
                log_lines.append(f"NO_DOI       {rel}")
                continue
        except Exception as e:
            counts["FETCH_FAIL"] += 1
            log_lines.append(f"FETCH_FAIL   {rel}  ref={doi or arxiv}  err={e}")
            continue

        try:
            parsed = parse_bibtex(bibtex)
        except Exception as e:
            counts["PARSE_FAIL"] += 1
            log_lines.append(f"PARSE_FAIL   {rel}  ref={doi or arxiv}  err={e}")
            continue
        fields = parsed["fields"]

        missing = [f for f in ("author", "title") if not fields.get(f)]
        if not (fields.get("year") or fields.get("date")):
            missing.append("year")
        if missing:
            counts["INCOMPLETE"] += 1
            log_lines.append(
                f"INCOMPLETE   {rel}  ref={doi or arxiv}  missing={missing}"
            )
            continue

        # DOI roundtrip (only meaningful if we had a DOI to begin with)
        bib_doi = fields.get("doi", "").strip()
        if doi and bib_doi and bib_doi.lower() != doi.lower():
            counts["DOI_DIVERGE"] += 1
            log_lines.append(
                f"DOI_DIVERGE  {rel}  requested={doi}  in_bibtex={bib_doi}"
            )

        # Title overlap
        overlap = title_token_overlap(title or "", fields.get("title", ""))
        if overlap < TITLE_OVERLAP_THRESHOLD:
            counts["TITLE_MISMATCH"] += 1
            log_lines.append(
                f"TITLE_MISMATCH  {rel}  overlap={overlap:.2f}  "
                f"briefer={title!r}  bibtex={fields.get('title')!r}"
            )

        new_key = make_citekey(fields)
        formatted = format_bibtex(parsed["type"], new_key, fields)
        base = new_key
        suffix = 1
        while new_key in entries and entries[new_key] != formatted:
            suffix += 1
            new_key = f"{base}{chr(ord('a') + suffix - 2)}"
            formatted = format_bibtex(parsed["type"], new_key, fields)

        entries[new_key] = formatted
        counts["OK"] += 1
        inject_result = inject_citekey_into_briefer(briefer, new_key)
        log_lines.append(
            f"OK           {rel}  ref={doi or arxiv}  key={new_key}  "
            f"inject={inject_result}  src={source}"
        )
        time.sleep(args.sleep)

    sorted_entries = [entries[k] for k in sorted(entries.keys())]
    header = (
        "% Generated by .claude/skills/briefer-bibliography/build_bib.py\n"
        "% Source: per-paper briefer .md files in writeup/literature/\n"
        "% Auto-generated entries are above the MANUAL ENTRIES sentinel;\n"
        "% hand-maintained entries (data products, datasets, grey literature)\n"
        "% go below the sentinel and are preserved across regenerations.\n\n"
    )
    # Preserve any manual entries appended below the sentinel line.
    manual_section = ""
    if out_path.exists():
        old = out_path.read_text(encoding="utf-8")
        idx = old.find(MANUAL_SENTINEL)
        if idx != -1:
            manual_section = "\n\n" + old[idx:].rstrip() + "\n"
    auto_block = header + "\n\n".join(sorted_entries) + "\n"
    if not manual_section:
        # First run after sentinel introduction: append an empty sentinel so
        # future hand-edits below it survive.
        manual_section = "\n\n" + MANUAL_SENTINEL + "\n"
    out_path.write_text(auto_block + manual_section, encoding="utf-8")
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    print()
    print(f"Wrote {out_path} with {counts['OK']} entries.")
    print(f"  no DOI/arXiv in briefer  : {counts['NO_DOI']}")
    print(f"  fetch failures           : {counts['FETCH_FAIL']}")
    print(f"  parse failures           : {counts['PARSE_FAIL']}")
    print(f"  incomplete metadata      : {counts['INCOMPLETE']}")
    print(f"  DOI divergence warnings  : {counts['DOI_DIVERGE']}")
    print(f"  title-mismatch warnings  : {counts['TITLE_MISMATCH']}")
    print(f"Log: {log_path}")

    if args.strict and (counts["NO_DOI"] or counts["FETCH_FAIL"]
                        or counts["PARSE_FAIL"] or counts["INCOMPLETE"]):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
