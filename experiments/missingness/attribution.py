"""Where do the dense networks look when September is missing? Gradient attribution per month on
holdout pixels for the month-dropout x gate arms of TempCNN and CNN-BiLSTM
(experiments/sparse_gate_backbones/models/{bb}_{none,none_md0.2,sparsemax,sparsemax_md0.2}).

Pixels: 5,000 September-missing and 5,000 September-observed holdout pixels (seed 42). Each
missing pixel is scored twice, with September zero-filled (the published inputs) and filled
with the training median; observed pixels are identical under both fills.

Per pixel and seed, for the class the model predicts:
  ig_share   share of total |integrated gradient| (128 steps, baseline = training mean, i.e. 0
             in TempCNN's standardized space and the raw training mean for CNN-BiLSTM) that falls
             on each month, summed over the six bands
  grad_share share of total |d logit / d x| per month (local sensitivity)
Step count is chosen per model (arm and seed): starting at 128, doubled (up to 2,048) until the completeness error
(|sum of IG - logit change| / |logit change|) on 1,024 evenly spaced zero-filled missing pixels has a median
below 2%. Zero-filled inputs lie far from the baseline and the sparsemax gate is sharp, so 32
steps gave 6% (dense TempCNN) and 128 steps 5.5% (gated TempCNN). Every model, seed and input set
must then reach a median error below 3%; steps and errors are recorded. Finished arms are cached
in results/attribution_cache/ so a restart skips them.
Shares are averaged over pixels and then over the 5 seeds. Also: how often the 5-seed ensemble's
pixel prediction changes between the two fills (September-missing pixels only).

Writes results/attribution_by_month.csv, results/attribution_september.csv,
figures/attribution_september.{pdf,png}. Run in the deep_field env:
  python experiments/missingness/attribution.py
"""
import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
SG = os.path.join(REPO, "experiments", "sparse_gate_backbones")
sys.path.insert(0, SG)
import train as tr  # noqa: E402

N_PIX, BS, SEED = 5_000, 4096, 42
STEP_GRID = (128, 256, 512, 1024, 2048)
SEPT = tr.MONTHS.index("September")
ARMS = ["none", "none_md0.2", "sparsemax", "sparsemax_md0.2"]
ARM_LABEL = {"none": "dense", "none_md0.2": "dense + dropout", "sparsemax": "gated",
             "sparsemax_md0.2": "gated + dropout"}


def sample_rows():
    """Row indices of September-missing and -observed holdout pixels (same rows in both files)."""
    a = pq.read_table(tr.HOLDOUT["orig"], columns=["fid", "B2_September"]).to_pandas()
    b = pq.read_table(tr.HOLDOUT["trainmed"], columns=["fid"]).to_pandas()
    assert (a["fid"].values == b["fid"].values).all(), "holdout files are not row-aligned"
    rng = np.random.default_rng(SEED)
    miss = a.index[a["B2_September"].isna()].values
    obs = a.index[a["B2_September"].notna()].values
    return {"missing": np.sort(rng.choice(miss, N_PIX, replace=False)),
            "observed": np.sort(rng.choice(obs, N_PIX, replace=False))}


def holdout_inputs(path, rows, art, layout):
    """train.load_holdout on a subset of rows: missing columns 0, NaN -> 0, training scaler."""
    df = pq.read_table(path, columns=[c for c in art["cols"] if c in pq.ParquetFile(path).schema.names]
                       ).to_pandas().iloc[rows]
    for c in art["cols"]:
        if c not in df.columns:
            df[c] = 0.0
    X = df[art["cols"]].fillna(0).values
    if art["scaler"] is not None:
        X = np.nan_to_num(art["scaler"].transform(X), nan=0.0, posinf=0.0, neginf=0.0)
    return tr.to_seq(X.astype(np.float32), layout)


def baseline(art, layout):
    if art["scaler"] is not None:
        return np.zeros((tr.T, tr.C), np.float32)  # training mean in standardized space
    X = pq.read_table(tr.MERGED_DL_PATH, columns=art["cols"]).to_pandas().fillna(0).values
    return tr.to_seq(X.mean(0, keepdims=True).astype(np.float32), layout)[0]


def attributions(model, X, base, cls, steps):
    """Integrated gradients and plain gradients of the logit of class `cls`, per pixel."""
    ig = np.zeros_like(X); gr = np.zeros_like(X)
    b = torch.from_numpy(base).to(tr.device)
    with torch.backends.cudnn.flags(enabled=False):  # cuDNN RNN backward needs train mode
        for i in range(0, len(X), BS):
            x = torch.from_numpy(X[i:i + BS]).to(tr.device)
            c = torch.from_numpy(cls[i:i + BS]).to(tr.device)
            acc = torch.zeros_like(x)
            for alpha in (np.arange(steps) + 0.5) / steps:  # midpoint rule
                z = (b + alpha * (x - b)).requires_grad_(True)
                out = model(z).gather(1, c[:, None]).sum()
                g, = torch.autograd.grad(out, z)
                acc += g
            ig[i:i + BS] = ((x - b) * acc / steps).cpu().numpy()
            z = x.clone().requires_grad_(True)
            g, = torch.autograd.grad(model(z).gather(1, c[:, None]).sum(), z)
            gr[i:i + BS] = g.cpu().numpy()
    return ig, gr


def completeness(model, X, base, ig, lg, cls):
    with torch.no_grad():
        fb = model(torch.from_numpy(base[None]).to(tr.device))[0].cpu().numpy()
    gap = lg.numpy()[np.arange(len(cls)), cls] - fb[cls]
    return float(np.median(np.abs(ig.sum((1, 2)) - gap) / np.maximum(np.abs(gap), 1e-6)))


def calibrate(model, X, base):
    X = X[np.linspace(0, len(X) - 1, 1024).astype(int)]  # spread over the sample (rows are sorted by position)
    lg, _ = tr.logits_of(model, X, BS)
    cls = lg.argmax(1).numpy()
    for steps in STEP_GRID:
        err = completeness(model, X, base, attributions(model, X, base, cls, steps)[0], lg, cls)
        if err < 0.02:
            break
    return steps, err


def month_share(A):
    m = np.abs(A).sum(2)  # (N, T)
    return m / np.maximum(m.sum(1, keepdims=True), 1e-12)


def load_model(bb, arm, nc, seed):
    gate = arm.split("_")[0]
    m = (tr.TempCNN if bb == "tempcnn" else tr.CNNBiLSTM)(tr.C, nc) if gate == "none" else tr.Gated(bb, gate, nc)
    m.load_state_dict(torch.load(f"{SG}/models/{bb}_{arm}/seed_{seed}.pt", map_location=tr.device))
    return m.to(tr.device).eval()


def main():
    rows = sample_rows()
    by_month, sept = [], []
    cache = os.path.join(HERE, "results", "attribution_cache")
    os.makedirs(cache, exist_ok=True)
    for bb in ["tempcnn", "cnn_bilstm"]:
        r = tr.RECIPE[bb]
        for arm in ARMS:
            cfile = os.path.join(cache, f"{bb}_{arm}.joblib")
            if os.path.exists(cfile):
                bm_rows, sp_rows = joblib.load(cfile)
                for row in sp_rows:  # arms cached before per-seed calibration (same 3% check passed)
                    if "ig_steps" in row:
                        row["ig_steps_max"] = row.pop("ig_steps")
                by_month += bm_rows; sept += sp_rows
                print(f"{bb} {arm}: cached", flush=True)
                continue
            n_bm, n_sp = len(by_month), len(sept)
            art = joblib.load(f"{SG}/models/{bb}_{arm}/artifacts.joblib")
            nc = len(art["le"].classes_)
            base = baseline(art, r["layout"])
            inputs = {("missing", "zero"): holdout_inputs(tr.HOLDOUT["orig"], rows["missing"], art, r["layout"]),
                      ("missing", "train_median"): holdout_inputs(tr.HOLDOUT["trainmed"], rows["missing"], art, r["layout"]),
                      ("observed", "as observed"): holdout_inputs(tr.HOLDOUT["orig"], rows["observed"], art, r["layout"])}
            assert not np.allclose(inputs[("missing", "zero")][:, SEPT], inputs[("missing", "train_median")][:, SEPT])
            shares = {k: {"ig": [], "grad": []} for k in inputs}
            logits = {k: [] for k in inputs}
            errs, used = [], []
            for seed in r["seeds"]:
                model = load_model(bb, arm, nc, seed)
                steps, cal = calibrate(model, inputs[("missing", "zero")], base)
                used.append(steps)
                print(f"{bb} {arm} seed {seed}: {steps} IG steps (calibration error {cal:.3f})", flush=True)
                for k, X in inputs.items():
                    lg, _ = tr.logits_of(model, X, BS)
                    cls = lg.argmax(1).numpy()
                    logits[k].append(lg)
                    ig, gr = attributions(model, X, base, cls, steps)
                    err = completeness(model, X, base, ig, lg, cls)  # sum of IG = f(x) - f(baseline)
                    errs.append(err)
                    assert err < 0.03, f"{bb}_{arm} seed {seed} {k}: IG completeness error {err:.3f}"
                    shares[k]["ig"].append(month_share(ig).mean(0))
                    shares[k]["grad"].append(month_share(gr).mean(0))
                del model; torch.cuda.empty_cache()
            pred = {k: torch.stack(v).mean(0).argmax(1).numpy() for k, v in logits.items()}
            flip = float(np.mean(pred[("missing", "zero")] != pred[("missing", "train_median")]))
            for (pix, fill), s in shares.items():
                ig, gr = np.mean(s["ig"], 0), np.mean(s["grad"], 0)
                for t, mo in enumerate(tr.MONTHS):
                    by_month.append({"backbone": bb, "arm": ARM_LABEL[arm], "pixels": pix, "fill": fill,
                                     "month": mo, "ig_share": ig[t], "grad_share": gr[t]})
                sept.append({"backbone": bb, "arm": ARM_LABEL[arm], "pixels": pix, "fill": fill,
                             "ig_share_sept": ig[SEPT], "ig_share_sept_seed_sd": np.std([x[SEPT] for x in s["ig"]]),
                             "grad_share_sept": gr[SEPT],
                             "pred_flip_zero_vs_median": flip if pix == "missing" else np.nan,
                             "ig_steps_max": max(used), "ig_completeness_err_max": max(errs)})
            joblib.dump((by_month[n_bm:], sept[n_sp:]), cfile)
            print(f"{bb:10s} {ARM_LABEL[arm]:16s} IG Sept share: missing/zero "
                  f"{sept[-3]['ig_share_sept']:.3f}  missing/median {sept[-2]['ig_share_sept']:.3f}  "
                  f"observed {sept[-1]['ig_share_sept']:.3f}  flips {flip:.3f}", flush=True)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    bm, sp = pd.DataFrame(by_month).round(4), pd.DataFrame(sept).round(4)
    bm.to_csv(os.path.join(HERE, "results", "attribution_by_month.csv"), index=False)
    sp.to_csv(os.path.join(HERE, "results", "attribution_september.csv"), index=False)
    figure(bm)


def figure(bm):
    sys.path.insert(0, os.path.join(REPO, "writeup", "figures"))
    from figstyle import apply_style
    apply_style()
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.6), sharey=True, sharex=True)
    x = np.arange(len(tr.MONTHS))
    style = {("missing", "zero"): ("#d55e00", "-", "o", "Sept missing, zero fill"),
             ("missing", "train_median"): ("#0072b2", "-", "s", "Sept missing, training-median fill"),
             ("observed", "as observed"): ("#555555", "--", None, "Sept observed")}
    for i, bb in enumerate(["tempcnn", "cnn_bilstm"]):
        for j, arm in enumerate(ARMS):
            ax = axes[i, j]
            d = bm[(bm.backbone == bb) & (bm.arm == ARM_LABEL[arm])]
            for (pix, fill), (col, ls, mk, lab) in style.items():
                y = d[(d.pixels == pix) & (d.fill == fill)].set_index("month").loc[tr.MONTHS, "ig_share"]
                ax.plot(x, y, color=col, ls=ls, marker=mk, ms=3.5, lw=1.4, label=lab)
            ax.axvline(SEPT, color="black", lw=0.5, alpha=0.3)
            ax.set_title(f"({chr(97 + 4 * i + j)}) {'TempCNN' if bb == 'tempcnn' else 'CNN-BiLSTM'}, "
                         f"{ARM_LABEL[arm]}", fontsize=9.5)
            ax.set_xticks(x, [m[:3] for m in tr.MONTHS], fontsize=7, rotation=90)
        axes[i, 0].set_ylabel("Share of |integrated gradient|", fontsize=9)
    axes[0, 0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, "figures", f"attribution_september.{ext}"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
