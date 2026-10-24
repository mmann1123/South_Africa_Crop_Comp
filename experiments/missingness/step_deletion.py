"""Step deletion: the attention models classify each holdout pixel from its observed months only.

Instead of filling a missing month, its time step is masked out of temporal attention (logit set
to -inf before the softmax, so its weight is exactly 0 and the remaining weights renormalize) and,
for the Transformer, out of self-attention (key-padding mask) and the final mean over time. A
month counts as missing for a pixel if any of its six bands is NaN, so no value is imputed
anywhere. Month positions are fixed per slot (MONTH_POSITIONS), so masking needs no other change.

Inference only, on existing checkpoints (no retraining):
  L-TAE (published, deep_learn/src/models), L-TAE-S (field_reduction/models/ltae_sparse_pixel/
  frac_1.00), Transformer (published), and the month-dropout L-TAE / L-TAE-S
  (month_dropout/models/*_md0.2). Same 5-seed logit averaging, AMP and field majority vote as the
  published inference.

Checks (abort on failure): masked forward with every step observed equals model(x); masked output
does not depend on the value in a masked step; with nothing masked under zero fill, field
predictions reproduce the existing zero-fill predictions.

Writes predictions/step_deletion/{model}_stepdel.csv and results/step_deletion.csv (macro-F1 with
paired field-bootstrap CIs vs zero and training-median fill of the same checkpoints; B = 10,000,
seed 42). Run in the deep_field env: python experiments/missingness/step_deletion.py
"""
import math
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
import torch
from joblib import load

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in ("deep_learn/src", "out_of_sample", "experiments/field_reduction"):
    sys.path.insert(0, os.path.join(REPO, p))
from config import MERGED_DL_TEST_PATH, MODEL_DIR  # noqa: E402
from compare_predictions import load_ground_truth  # noqa: E402
from paired_bootstrap_gaps import confusion, metrics  # noqa: E402
import inference_ltae as il  # noqa: E402
import inference_transformer as it  # noqa: E402
import models_arch as ma  # noqa: E402
import predict_oos as po  # noqa: E402

device = torch.device("cuda")
SEEDS = [42, 101, 202, 303, 404]
T, C = 10, 6
FILL = os.path.join(REPO, "experiments", "holdout_fill_check", "predictions")
MD = os.path.join(HERE, "month_dropout")
OUT = os.path.join(HERE, "predictions", "step_deletion")
B, BOOT_SEED = 10_000, 42


# ------------------------------------------------------------------ masked forwards
def ltae_masked(m, x, obs):
    """inference_ltae.LTAE / models_arch.LTAE forward with steps where obs is False removed."""
    B_, T_, _ = x.shape
    x = m.pos_enc(m.embedding(x))
    keys = m.key_proj(x).view(B_, T_, m.n_head, m.d_k).permute(2, 0, 1, 3)
    values = m.value_proj(x).view(B_, T_, m.n_head, m.d_k).permute(2, 0, 1, 3)
    query = m.query.unsqueeze(1).unsqueeze(2).expand(-1, B_, -1, -1)
    attn = torch.matmul(query, keys.transpose(-2, -1)) / math.sqrt(m.d_k)  # (H, B, 1, T)
    attn = attn.masked_fill(~obs[None, :, None, :], float("-inf"))
    attn = m.attention_dropout(torch.softmax(attn, dim=-1))
    out = torch.matmul(attn, values).squeeze(2).permute(1, 0, 2).reshape(B_, -1)
    return m.classifier(m.mlp(m.norm(out)))


def ltae_s_masked(m, x, obs):
    """models_arch.LTAESparse forward with masked steps (the gate acts per step, so it is unchanged)."""
    B_ = x.shape[0]
    x = m.pos_enc(m.embedding(x))
    masked_xs, _ = m.feature_gate(x)
    heads = []
    for h in range(m.n_head):
        k = m.key_projs[h](masked_xs[h]); v = m.value_projs[h](masked_xs[h])
        q = m.query[h].unsqueeze(0).unsqueeze(0).expand(B_, -1, -1)
        a = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(m.d_k)  # (B, 1, T)
        a = m.attention_dropout(torch.softmax(a.masked_fill(~obs[:, None, :], float("-inf")), dim=-1))
        heads.append(torch.matmul(a, v).squeeze(1))
    return m.classifier(m.mlp(m.norm(torch.cat(heads, dim=-1))))


def transformer_masked(m, x, obs):
    """inference_transformer.SITSTransformer with a key-padding mask and a mean over observed steps."""
    x = m.pos_enc(m.embedding(x))
    x = m.encoder_norm(m.encoder(x, src_key_padding_mask=~obs))
    w = obs.to(x.dtype)[..., None]
    x = (torch.nan_to_num(x) * w).sum(1) / w.sum(1)
    return m.classifier(m.mlp(x))


# ------------------------------------------------------------------ models
def published_ltae():
    d = MODEL_DIR
    art = (load(f"{d}/ltae_feature_columns.joblib"), load(f"{d}/ltae_scaler.joblib"), load(f"{d}/ltae_label_encoder.joblib"))
    mk = lambda s, nc: il.LTAE(in_channels=C, num_classes=nc), f"{d}/ltae_seed_{{}}.pt"  # noqa: E731
    return art, mk


def fr_ltae(d, sparse):
    art = (load(f"{d}/feature_columns.joblib"), load(f"{d}/scaler.joblib"), load(f"{d}/label_encoder.joblib"))
    if sparse:
        return art, (lambda s, nc: ma.LTAESparse(in_channels=C, num_classes=nc, **po._ltae_s_arch(d)),
                     f"{d}/ltae_sparse_seed_{{}}.pt")
    return art, (lambda s, nc: ma.LTAE(in_channels=C, num_classes=nc), f"{d}/ltae_seed_{{}}.pt")


def published_transformer():
    d = MODEL_DIR
    art = (load(f"{d}/transformer_feature_columns.joblib"), load(f"{d}/transformer_scaler.joblib"),
           load(f"{d}/transformer_label_encoder.joblib"))
    return art, (lambda s, nc: it.SITSTransformer(in_channels=C, num_classes=nc), f"{d}/transformer_seed_{{}}.pt")


FR = os.path.join(REPO, "experiments", "field_reduction", "models")
# name -> (loader, masked forward, plain forward returns logits only?, reference stem for the fills)
MODELS = {
    "ltae": (published_ltae, ltae_masked, True, f"{FILL}/ltae_{{}}.csv"),
    "ltae_s": (lambda: fr_ltae(f"{FR}/ltae_sparse_pixel/frac_1.00", True), ltae_s_masked, False, f"{FILL}/ltae_s_{{}}.csv"),
    "transformer": (published_transformer, transformer_masked, True, f"{FILL}/transformer_{{}}.csv"),
    "ltae_md0.2": (lambda: fr_ltae(f"{MD}/models/ltae_pixel_md0.2", False), ltae_masked, True,
                   f"{MD}/predictions/ltae_pixel_md0.2_{{}}.csv"),
    "ltae_s_md0.2": (lambda: fr_ltae(f"{MD}/models/ltae_sparse_pixel_md0.2", True), ltae_s_masked, False,
                     f"{MD}/predictions/ltae_sparse_pixel_md0.2_{{}}.csv"),
}


def plain(m, x, logits_only):
    out = m(x)
    return out if logits_only else out[0]


def run(name, raw, fids):
    loader, masked, logits_only, _ = MODELS[name]
    (cols, scaler, le), (make, path) = loader()
    df = raw[cols]
    month_missing = df.isna().values.reshape(-1, T, C).any(2)  # (N, T)
    X = np.nan_to_num(scaler.transform(df.fillna(0).values).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    X = X.reshape(-1, T, C)
    obs_all = torch.from_numpy(~month_missing)
    none_obs = (~obs_all).all(1)
    logit = {"stepdel": [], "zero": []}
    for s in SEEDS:
        m = make(s, len(le.classes_)).to(device)
        m.load_state_dict(torch.load(path.format(s), map_location=device, weights_only=True))
        m.eval()
        with torch.no_grad(), torch.amp.autocast("cuda"):
            # checks on one batch
            xb = torch.from_numpy(X[:4096]).to(device)
            full = torch.ones(xb.shape[:2], dtype=torch.bool, device=device)
            assert torch.allclose(masked(m, xb, full).float(), plain(m, xb, logits_only).float(), atol=2e-2), \
                f"{name}: masked forward with all steps observed differs from model(x)"
            part = full.clone(); part[:, 6] = False  # September
            xb2 = xb.clone(); xb2[:, 6] += 5.0
            assert torch.allclose(masked(m, xb, part).float(), masked(m, xb2, part).float(), atol=1e-3), \
                f"{name}: output depends on a masked step"
            out = {k: [] for k in logit}
            for i in range(0, len(X), 4096):
                xb = torch.from_numpy(X[i:i + 4096]).to(device)
                ob = obs_all[i:i + 4096].to(device)
                ob = ob | (~ob).all(1, keepdim=True)  # no observed month: fall back to the zero-filled input
                out["stepdel"].append(masked(m, xb, ob).float().cpu())
                out["zero"].append(plain(m, xb, logits_only).float().cpu())
        for k in logit:
            logit[k].append(torch.cat(out[k]))
        del m; torch.cuda.empty_cache()
    res = {}
    for k, v in logit.items():
        p = torch.stack(v).mean(0).argmax(1).numpy()
        fp = pd.DataFrame({"fid": fids, "pred": p}).groupby("fid")["pred"].agg(lambda x: Counter(x).most_common(1)[0][0])
        res[k] = pd.Series(le.inverse_transform(fp.values), index=fp.index, name="crop_name")
    return res, month_missing, int(none_obs.sum())


def main():
    os.makedirs(OUT, exist_ok=True)
    raw = pd.read_parquet(MERGED_DL_TEST_PATH)
    fids = raw["fid"].values
    gt = load_ground_truth().drop_duplicates("fid").set_index("fid")["true_label"]
    cl = sorted(gt.unique()); ci = {c: i for i, c in enumerate(cl)}; K = len(cl)
    rng = np.random.default_rng(BOOT_SEED)
    rows = []
    for name in MODELS:
        res, mm, n_none = run(name, raw, fids)
        res["stepdel"].reset_index().to_csv(f"{OUT}/{name}_stepdel.csv", index=False)
        ref = MODELS[name][3]
        zero_ref = pd.read_csv(ref.format("orig")).drop_duplicates("fid").set_index("fid")["crop_name"]
        n_diff = int((res["zero"].reindex(zero_ref.index) != zero_ref).sum())
        assert n_diff <= 2, f"{name}: zero-fill rerun differs from existing predictions on {n_diff} fields"
        preds = {"step deletion": res["stepdel"], "zero fill": zero_ref,
                 "training-median fill": pd.read_csv(ref.format("trainmed")).drop_duplicates("fid").set_index("fid")["crop_name"]}
        idx = gt.index
        for p in preds.values():
            idx = idx.intersection(p.index)
        yt = gt.reindex(idx).map(ci).values
        yp = {k: p.reindex(idx).map(ci).values for k, p in preds.items()}
        N = len(idx)
        W = [np.bincount(rng.integers(0, N, N), minlength=N).astype(float) for _ in range(B)]
        f = {k: metrics(confusion(yt, y, K))[0] for k, y in yp.items()}
        bs = {k: np.array([metrics(confusion(yt, y, K, w))[0] for w in W]) for k, y in yp.items()}
        r = {"model": name, "pixels_with_a_deleted_step_pct": round(100 * mm.any(1).mean(), 2),
             "pixels_all_steps_missing": n_none, "zero_rerun_fields_differing": n_diff}
        for k in preds:
            r[f"f1m_{k}"] = f[k]
        for b in ("zero fill", "training-median fill"):
            lo, hi = np.percentile(bs["step deletion"] - bs[b], [2.5, 97.5])
            r[f"stepdel_minus_{b}"] = f["step deletion"] - f[b]
            r[f"ci_lo_vs_{b}"], r[f"ci_hi_vs_{b}"] = lo, hi
        rows.append(r)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    pd.DataFrame(rows).round(4).to_csv(os.path.join(HERE, "results", "step_deletion.csv"), index=False)


if __name__ == "__main__":
    main()
