#!/usr/bin/env python
"""Evaluate the released Sen4AgriNet PAD checkpoints (U-Net, ConvLSTM, ConvSTAR)
on their published test splits under test-time interventions on the monthly
inputs, and store one confusion matrix per (scenario, model, patch, condition)
under both the authors' test metric and a corrected one (evalcore.py).

Conditions (rerunlib.model_inputs builds every one with the authors' median and
fill rule, verified equal to their code):
  a           as published
  del_m{0..11} delete every acquisition of month m (0 = January) when the
              patch has one; the month is then filled by the authors' rule.
              All twelve months, because the released checkpoints consume B03
              and B04 for every month (rerunlib.export_slots), not the
              April to September window the paper describes
  src_{k}     impose the empty-month pattern of the k-th randomly drawn
              training-domain patch (k = 0..4): its empty months are deleted
              from this patch (months this patch already lacks stay filled)
  tgt_{k}     the same with a pattern drawn from the test domain
  rnd_{k}     delete as many months as src_{k} deleted, chosen at random among
              the months this patch has
  cloud       per-acquisition Hollstein cloud mask (the authors' own
              utils/tools.hollstein_mask, clouds only) before the median, so
              cloudy pixel-months fall back to the fill
Scenario 1 is run with condition a only (the in-domain bound).

Patterns are drawn from the descriptive audit's sampled time stacks
(../sen4agrinet/results/patch_time_stacks.csv.gz, 25 patches per scenario,
split and tile-year). Seed 42 for the draws; the draws for a patch depend only
on (seed, scenario, stem), so they are identical across models and resumes.
Input building runs in a process pool; one GPU process runs the models.
Resumable per scenario and chunk of patches.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.external_audits.common import paths  # noqa: E402
from experiments.external_audits.common.runlog import RunLog  # noqa: E402
from experiments.external_audits.sen4agrinet import splits  # noqa: E402
from experiments.external_audits.sen4agrinet_rerun import evalcore as ec  # noqa: E402
from experiments.external_audits.sen4agrinet_rerun import rerunlib as rl  # noqa: E402

PAPER = "sen4agrinet_rerun"
ARCHS = ["unet", "convlstm", "convstar"]
NETCDF = paths.paper_root(PAPER) / "netcdf"
S4A_RES = paths.AUDIT_DIR / "sen4agrinet" / "results"
N_DRAWS = 5
LAYOUT = "export"  # what the released checkpoints were trained on
K = 12
_CFG = None
_PATTERNS = None


def month_sets_from_stacks() -> pd.DataFrame:
    """Per sampled patch: scenario, split, stem, and the set of months (0..11) with data."""
    st = pd.read_csv(S4A_RES / "patch_time_stacks.csv.gz")
    smp = pd.read_csv(S4A_RES / "patch_sample.csv")
    rows = []
    for _, r in st.iterrows():
        year = int(r["stem"][:4])
        t = pd.DatetimeIndex(pd.to_datetime(str(r["times"]).split(";")))
        m = rl.month_of(t, year)
        rows.append({"stem": r["stem"], "present": frozenset(int(x) for x in m[m >= 0])})
    ms = pd.DataFrame(rows)
    return smp.merge(ms, on="stem", how="inner")


def draw_rng(scenario: int, stem: str) -> np.random.Generator:
    h = int(hashlib.sha256(f"42|{scenario}|{stem}".encode()).hexdigest()[:16], 16)
    return np.random.default_rng(h)


def plan_for(scenario: int, stem: str, present: set[int], full: bool) -> list[tuple[str, int, set[int] | str]]:
    """List of (condition, draw, months_to_delete or 'cloud')."""
    out = [("a", -1, set())]
    if not full:
        return out
    for m in range(12):
        if m in present:
            out.append((f"del_m{m}", -1, {m}))
    rng = draw_rng(scenario, stem)
    src = _PATTERNS[(scenario, "train")]
    tgt = _PATTERNS[(scenario, "test")]
    for k in range(N_DRAWS):
        ps = src[rng.integers(len(src))]
        pt = tgt[rng.integers(len(tgt))]
        del_s = set(present) - set(ps)
        del_t = set(present) - set(pt)
        pres = sorted(present)
        n = min(len(del_s), max(len(pres) - 2, 0))
        del_r = set(rng.choice(pres, size=n, replace=False).tolist()) if n > 0 else set()
        out += [(f"src_{k}", k, del_s), (f"tgt_{k}", k, del_t), (f"rnd_{k}", k, del_r)]
    out.append(("cloud", -1, "cloud"))
    return out


def _init(patterns):
    global _CFG, _PATTERNS
    _CFG = ec.load_config()
    _PATTERNS = patterns
    torch.set_num_threads(1)


def build(job):
    scenario, stem, full = job
    path = NETCDF / splits.hf_path(stem)
    bands = sorted(set(rl.MODEL_BANDS + (rl.HOLLSTEIN_BANDS if full else [])))
    p = rl.read_patch(path, bands)
    months = rl.month_of(p.times, p.year)
    present = set(int(x) for x in np.unique(months[months >= 0]))
    items = []
    for cond, draw, spec in plan_for(scenario, stem, present, full):
        if spec == "cloud":
            x = rl.model_inputs(p, _CFG.LINEAR_ENCODER, pixel_ok=rl.hollstein_ok(p), layout=LAYOUT)
            ndel = 0
        else:
            keep = ~np.isin(months, list(spec)) if spec else None
            x = rl.model_inputs(p, _CFG.LINEAR_ENCODER, keep=keep, layout=LAYOUT)
            ndel = len(spec)
        items.append({"cond": cond, "draw": draw, "n_deleted": ndel, "deleted": ",".join(map(str, sorted(spec))) if isinstance(spec, set) else spec,
                      "medians": x["medians"], "labels": x["labels"], "parcels": x["parcels"],
                      "months_filled": int((x["present_share"] < 0.5).sum())})
    window_complete = len(present) == 12
    return scenario, stem, sorted(present), window_complete, items


def bounded(pool, jobs, max_in_flight: int):
    """Yield results as they finish, with at most max_in_flight patches built
    and not yet consumed (unbounded imap let 16 workers outrun the GPU and the
    queued inputs, about 375 MB per patch, exhausted memory)."""
    from collections import deque  # noqa: PLC0415

    pending, it = deque(), iter(jobs)
    for job in it:
        pending.append(pool.apply_async(build, (job,)))
        if len(pending) >= max_in_flight:
            break
    while pending:
        yield pending.popleft().get()
        nxt = next(it, None)
        if nxt is not None:
            pending.append(pool.apply_async(build, (nxt,)))


def run_scenario(scenario: int, full: bool, workers: int, chunk: int, limit: int | None, log: RunLog) -> None:
    out_dir = paths.results_dir(PAPER) / "raw"
    out_dir.mkdir(exist_ok=True)
    sp = pd.read_csv(S4A_RES / "split_patches.csv.gz")
    stems = sorted(sp.loc[(sp["scenario"] == scenario) & (sp["split"] == "test"), "stem"])
    if limit:  # smoke runs: first patches already on disk, written to raw_smoke/
        stems = [x for x in stems if (NETCDF / splits.hf_path(x)).exists()][:limit]
        out_dir = paths.results_dir(PAPER) / "raw_smoke"
        out_dir.mkdir(exist_ok=True)
    wroot = paths.paper_root(PAPER) / "dropbox"
    models = {a: ec.load_model(a, ec.checkpoint_path(wroot, a, scenario)) for a in ARCHS}
    patterns = {}
    ms = month_sets_from_stacks()
    for split in ("train", "test"):
        sel = ms[(ms["scenario"] == scenario) & (ms["split"] == split)]
        patterns[(scenario, split)] = [frozenset(x) for x in sel["present"]] or [frozenset(range(12))]
    for c0 in range(0, len(stems), chunk):
        tag = f"s{scenario}_{'full' if full else 'a'}_{c0:05d}"
        dest = out_dir / f"{tag}.parquet"
        if dest.exists():
            continue
        jobs = [(scenario, s, full) for s in stems[c0:c0 + chunk]]
        rows, t0 = [], time.time()
        with Pool(workers, initializer=_init, initargs=(patterns,)) as pool:
            for sc, stem, present, wc, items in bounded(pool, jobs, max_in_flight=workers + 8):
                for it in items:
                    x = torch.from_numpy(it["medians"]).cuda()
                    lab = torch.from_numpy(it["labels"]).cuda()
                    par = torch.from_numpy(it["parcels"]).cuda()
                    for a, m in models.items():
                        with torch.no_grad():
                            logp = ec.forward(m, a, x)
                        rows.append({"scenario": sc, "arch": a, "stem": stem, "cond": it["cond"], "draw": it["draw"],
                                     "n_deleted": it["n_deleted"], "deleted": it["deleted"], "months_filled": it["months_filled"],
                                     "window_complete": wc, "present": ",".join(map(str, present)),
                                     "cm_authors": ec.authors_confusion(logp, lab, par, K).ravel(),
                                     "cm_corrected": ec.corrected_confusion(logp, lab, par, K).ravel()})
        df = pd.DataFrame(rows)
        df["cm_authors"] = df["cm_authors"].map(lambda v: v.astype(np.int64).tolist())
        df["cm_corrected"] = df["cm_corrected"].map(lambda v: v.astype(np.int64).tolist())
        df.to_parquet(dest.with_suffix(".tmp"), index=False)
        dest.with_suffix(".tmp").replace(dest)
        print(f"{tag}: {len(jobs)} patches, {len(df)} rows, {(time.time() - t0) / 60:.1f} min", flush=True)
    log.add(f"scenario_{scenario}", {"patches": len(stems), "full": full})


def main(scenarios: list[int], full: bool, workers: int, chunk: int, limit: int | None) -> None:
    with RunLog(f"02_run_conditions_{'full' if full else 'a'}", {"scenarios": scenarios, "full": full, "workers": workers, "limit": limit}, paper=PAPER) as log:
        for s in scenarios:
            run_scenario(s, full and s != 1, workers, chunk, limit, log)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", type=int, nargs="+", default=[2, 3, 1])
    ap.add_argument("--full", action="store_true", help="all conditions (default: published inputs only, the gate)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--limit", type=int, default=None)
    main(**vars(ap.parse_args()))
