"""Completeness checks and deletion of downloaded external data (manifests kept)."""
from __future__ import annotations

import shutil
from pathlib import Path

from . import paths

# (relative path under results_dir(paper), minimum data rows)
EXPECTED: dict[str, list[tuple[str, int]]] = {
    "timematch": [("cloud_by_date.csv", 200), ("tile_summary.csv", 4), ("monthly_summary.csv", 40)],
    "denethor": [("s2_field_summary.csv", 1000), ("denethor_summary.csv", 4), ("pfqa_field_month.csv", 1000)],
    "daccn": [
        ("daccn_counters.csv|daccn_counters.csv.gz", 20000),
        ("daccn_summary.csv", 24),
        ("daccn_snow_counters.csv|daccn_snow_counters.csv.gz", 20000),
        ("daccn_snow_summary.csv", 24),
    ],
    "kerner": [("kerner_summary.csv", 96), ("kerner_pixel_states.csv", 9000), ("kerner_sample.csv", 9000)],
    "sen4agrinet": [("fill_by_split.csv", 8), ("patch_fill.csv", 600), ("oad_csv_crosscheck.csv", 16), ("rebuild_vs_earlier.csv", 8)],
    "stoian": [
        ("fig19_kappa_digitised.csv", 44),
        ("table1_class_areas.csv", 17),
        ("scl_tile_summary.csv", 11),
        ("l1c_tile_summary.csv", 11),
        ("stoian_tile_table.csv", 11),
        ("stoian_correlations.csv", 30),
    ],
    "gallo": [
        ("dates_by_area_year.csv", 600),
        ("date_counts_by_area_year.csv", 12),
        ("patch_footprints.csv", 90),
        ("cloud_by_date.csv", 680),
        ("pixel_state_raw.csv", 2000),
        ("gallo_area_year.csv", 12),
        ("gallo_pools.csv", 6),
    ],
    "wijesingha": [
        ("notebook_facts.csv", 18),
        ("cropland_points.csv", 4000),
        ("obs_counts_points.csv.gz", 36000),
        ("year_summary.csv", 18),
        ("period_contrast.csv", 2),
    ],
    "verhulst": [
        ("fig7_oa_digitised.csv", 75),
        ("forest_points.csv", 3000),
        ("valid_dekads_summary.csv", 15),
        ("dekad_valid_share.csv", 540),
        ("oa_loss_scenarios.csv", 180),
        ("oa_loss_correlations.csv", 100),
    ],
    "maleki": [
        ("site_inventory.csv", 4),
        ("interp_summary.csv", 30),
        ("monthly_valid.csv", 24),
        ("pixel_metrics.csv", 3000),
        ("maleki_direction_table.csv", 2),
    ],
    "hao": [
        ("fig_oa_digitised.csv", 108),
        ("points.csv", 10000),
        ("window_shares.csv", 400),
        ("domain_summary.csv", 20),
        ("prefix_summary.csv", 400),
        ("gap_vs_oa.csv", 100),
        ("hs_residual_cloud.csv", 60),
    ],
    "hexi": [
        ("points.csv", 2900),
        ("weekly_gap_summary.csv", 96),
        ("residual_cloud.csv", 24),
        ("weekly_valid_share.csv", 1200),
        ("monthly_qa60.csv", 72),
        ("discard_rules.csv", 24),
        ("qa60_images.csv", 100),
    ],
}
KEEP = ("manifest.json", "central_directory.json")


def expected_outputs(paper: str) -> list[tuple[Path, int]]:
    return [(Path(rel), n) for rel, n in EXPECTED[paper]]


def _count_rows(p: Path) -> int:
    import gzip

    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt") as fh:
        return max(0, sum(1 for _ in fh) - 1)


def complete(paper: str) -> tuple[bool, list[str]]:
    res = paths.results_dir(paper)
    problems = []
    for rel, min_rows in expected_outputs(paper):
        candidates = [res / name for name in rel.name.split("|")]
        found = next((c for c in candidates if c.exists()), None)
        if found is None:
            problems.append(f"missing {rel.name}")
            continue
        n = _count_rows(found)
        if n < min_rows:
            problems.append(f"{found.name}: {n} rows < {min_rows}")
    return (not problems), problems


def purge(paper: str, keep: tuple[str, ...] = KEEP, force: bool = False) -> int:
    ok, problems = complete(paper)
    if not ok and not force:
        raise RuntimeError(f"{paper} results incomplete, refusing to delete data: {problems}")
    root = paths.paper_root(paper)
    freed = 0
    for child in list(root.iterdir()):
        if child.name in keep:
            continue
        if child.is_dir():
            freed += sum(f.stat().st_size for f in child.rglob("*") if f.is_file())
            shutil.rmtree(child)
        else:
            freed += child.stat().st_size
            child.unlink()
    return freed
