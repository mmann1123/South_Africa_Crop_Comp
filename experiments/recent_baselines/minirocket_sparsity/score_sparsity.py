"""Score the MiniRocket sparsity variants against the reference MiniRocket and published models.

Reuses ../score.py (scoring, CIs, paired bootstrap + McNemar, provenance against
Table 1). Reference MiniRocket rows must also reproduce ../results/summary.csv.
Writes only to results/.

  ~/miniconda3/envs/ts_baselines/bin/python experiments/recent_baselines/minirocket_sparsity/score_sparsity.py
"""
import json
import os
import sys

import features as fx  # sets sys.path (parent experiment)
import common
import numpy as np
import pandas as pd

import score as base  # ../score.py

NEW = {  # display name -> model dir (this folder)
    "MiniRocket-S (pixel)": "minirocket_s_pixel",
    "MiniRocket-S (field)": "minirocket_s_field",
    "MiniRocket-softmax (pixel)": "minirocket_softmax_pixel",
    "MiniRocket-softmax (field)": "minirocket_softmax_field",
    "MiniRocket-L1 (pixel)": "minirocket_l1_pixel",
    "MiniRocket-L1 (field)": "minirocket_l1_field",
    "MiniRocket-ridge (pixel)": "minirocket_ridge_pixel",
    "MiniRocket-ridge (field)": "minirocket_ridge_field",
    "MiniRocket-ridge-ext (pixel)": "minirocket_ridge_ext_pixel",
    "MiniRocket-ridge-ext (field)": "minirocket_ridge_ext_field",
}
REF_MINIROCKET = {"MiniRocket (pixel)": "minirocket_pixel", "MiniRocket (field)": "minirocket_field"}
REF_PUBLISHED = ["L-TAE-S (pixel)", "L-TAE-S (field)", "L-TAE (pixel)", "L-TAE (field)",
                 "Logistic Regression (pixel)"]

PAIRS = []
for lv in ["pixel", "field"]:
    PAIRS += [
        (f"MiniRocket-S ({lv})", f"MiniRocket-softmax ({lv})", "sparsity itself (sparsemax vs softmax gate, same network)"),
        (f"MiniRocket-S ({lv})", f"MiniRocket ({lv})", "does the sparse gate help?"),
        (f"MiniRocket-softmax ({lv})", f"MiniRocket ({lv})", "does added capacity alone help?"),
        (f"MiniRocket-L1 ({lv})", f"MiniRocket ({lv})", "does global (L1) selection help?"),
        (f"MiniRocket-S ({lv})", f"L-TAE-S ({lv})", "gap to the paper's sparse model"),
    ]
PAIRS += [("MiniRocket-L1 (pixel)", "Logistic Regression (pixel)", "selected random features vs. raw features")]
for lv in ["pixel", "field"]:
    PAIRS += [
        (f"MiniRocket-ridge ({lv})", f"MiniRocket ({lv})", "canonical ridge head vs. L2 logistic head"),
        (f"MiniRocket-ridge-ext ({lv})", f"MiniRocket ({lv})", "ridge head (extended alpha grid) vs. L2 logistic head"),
        (f"MiniRocket-ridge-ext ({lv})", f"L-TAE-S ({lv})", "best-practice MiniRocket vs. the paper's sparse model"),
    ]
PAIRS += [("MiniRocket-ridge-ext (pixel)", "Logistic Regression (pixel)", "ridge on random features vs. logistic on raw features")]


def main():
    os.makedirs(fx.RESULTS_DIR, exist_ok=True)
    gt = base.ground_truth()
    pub_table = pd.read_csv(os.path.join(base.OOS, "scoring_results", "f1_macro_train_vs_oos.csv")).set_index("Model")
    parent_summary = pd.read_csv(os.path.join(common.RESULTS_DIR, "summary.csv")).set_index("model")

    csvs, in_region, meta = {}, {}, {}
    prov = []
    for name in REF_PUBLISHED:
        csv, key = base.PUBLISHED[name]
        csvs[name] = os.path.join(base.OOS, csv)
        in_region[name] = pub_table.loc[key, "Train F1 (macro)"]
        f1 = base.quiet_score(csvs[name])["f1_macro"]
        pub = pub_table.loc[key, "OOS F1 (macro)"]
        prov.append({"model": name, "f1m_here": round(f1, 4), "f1m_expected": round(pub, 4),
                     "source": "Table 1", "match": round(f1, 4) == round(pub, 4)})
    for name, d in REF_MINIROCKET.items():
        csvs[name] = os.path.join(common.PRED_DIR, f"{d}.csv")
        in_region[name] = json.load(open(os.path.join(common.MODELS_DIR, d, "metadata.json")))["metrics"]["f1_macro"]
        f1 = base.quiet_score(csvs[name])["f1_macro"]
        exp = parent_summary.loc[name, "oor_f1m"]
        prov.append({"model": name, "f1m_here": round(f1, 4), "f1m_expected": round(exp, 4),
                     "source": "../results/summary.csv", "match": round(f1, 4) == round(exp, 4)})
    prov = pd.DataFrame(prov)
    prov.to_csv(os.path.join(fx.RESULTS_DIR, "provenance_check.csv"), index=False)
    print(prov.to_string(index=False))
    if not prov["match"].all():
        sys.exit("Provenance check failed")

    for name, d in NEW.items():
        mp = os.path.join(fx.MODELS_DIR, d, "metadata.json")
        if not os.path.exists(mp):
            print(f"skip {name}: not trained yet")
            continue
        meta[name] = json.load(open(mp))
        csvs[name] = os.path.join(fx.PRED_DIR, f"{d}.csv")
        in_region[name] = meta[name]["metrics"]["f1_macro"]

    preds = {n: base.read_pred(c) for n, c in csvs.items()}
    classes = sorted(set(gt.unique()).union(*[set(p.unique()) for p in preds.values()]))
    cidx = {c: i for i, c in enumerate(classes)}
    K = len(classes)

    def aligned(*names):
        m = pd.concat([gt.rename("t")] + [preds[n].rename(n) for n in names], axis=1, join="inner")
        return [m["t"].map(cidx).values] + [m[n].map(cidx).values for n in names]

    rows = []
    for name in list(NEW) + list(REF_MINIROCKET) + REF_PUBLISHED:
        if name not in csvs:
            continue
        per_class = os.path.join(fx.RESULTS_DIR, "per_class", f"{NEW[name]}.csv") if name in meta else None
        s = base.quiet_score(csvs[name], per_class_out=per_class)
        yt, yp = aligned(name)
        lo, hi = base.single_ci(yt, yp, K, np.random.default_rng(base.SEED))
        row = {"model": name, "new": name in meta, "in_region_f1m": round(in_region[name], 4),
               "oor_f1m": round(s["f1_macro"], 4), "delta": round(s["f1_macro"] - in_region[name], 4),
               "ci_lo": round(lo, 4), "ci_hi": round(hi, 4), "kappa": round(s["kappa"], 4),
               "f1_weighted": round(s["f1_weighted"], 4), "accuracy": round(s["accuracy"], 4),
               "xent": round(s["xent"], 4)}
        if name in meta:
            stem = NEW[name]
            sf = [base.quiet_score(os.path.join(fx.PRED_DIR, f"{stem}_seed{sd}.csv"))["f1_macro"]
                  for sd in meta[name]["seeds"]]
            row.update({"seed_f1m_sd": round(float(np.std(sf, ddof=1)), 4),
                        "per_seed_train_seconds": meta[name]["per_seed_seconds_mean"]})
            ps = meta[name]["per_seed"]
            if "mask_stats_val" in ps[0]:
                for k in ["mean_nonzero_per_head", "mean_effective_features_per_head", "union_features_selected"]:
                    row[k] = round(float(np.mean([p["mask_stats_val"][k] for p in ps])), 1)
            if "nonzero_features_chosen" in ps[0]:
                row["l1_nonzero_features"] = round(float(np.mean([p["nonzero_features_chosen"] for p in ps])), 1)
                row["l1_chosen_C"] = ";".join(f"{p['chosen_C']:g}" for p in ps)
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(fx.RESULTS_DIR, "summary.csv"), index=False)
    print(summary.to_string(index=False))

    out = []
    for a, b, q in PAIRS:
        if a not in preds or b not in preds:
            continue
        yt, ya, yb = aligned(a, b)
        d, n10, n01, p_mcn = base.paired(yt, ya, yb, K, np.random.default_rng(base.SEED))
        pa, pb = np.array(base.metrics(base.confusion(yt, ya, K))), np.array(base.metrics(base.confusion(yt, yb, K)))
        for j, met in enumerate(["macro_f1", "kappa", "accuracy"]):
            lo, hi = np.percentile(d[:, j], [2.5, 97.5])
            p = min(1.0, 2 * min((d[:, j] <= 0).mean(), (d[:, j] >= 0).mean()))
            out.append({"A": a, "B": b, "question": q, "metric": met, "n_fields": len(yt),
                        "A_point": round(pa[j], 4), "B_point": round(pb[j], 4),
                        "delta": round(pa[j] - pb[j], 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                        "p_boot": round(p, 4), "sig_95": bool(lo > 0 or hi < 0),
                        "mcnemar_A_only_correct": n10, "mcnemar_B_only_correct": n01,
                        "mcnemar_p": float(f"{p_mcn:.3g}")})
    res = pd.DataFrame(out)
    res.to_csv(os.path.join(fx.RESULTS_DIR, "paired_bootstrap.csv"), index=False)
    if len(res):
        print(res[res.metric == "macro_f1"].drop(columns=["question", "n_fields", "metric"]).to_string(index=False))


if __name__ == "__main__":
    main()
