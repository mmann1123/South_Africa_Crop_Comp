"""Regenerate the tables in RESULTS.md from results/*.csv (hand-written text is kept).

  ~/miniconda3/envs/ts_baselines/bin/python experiments/recent_baselines/minirocket_sparsity/make_report.py
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from make_report import BEGIN, END, fmt_time, md  # noqa: E402  (../make_report.py)

RES = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "RESULTS.md")


def generated():
    parts = []
    s = pd.read_csv(os.path.join(RES, "summary.csv"))
    col = lambda c: s[c] if c in s else pd.Series([float("nan")] * len(s))
    num = lambda v, f: "" if pd.isna(v) else f.format(v)
    t = pd.DataFrame({
        "Model": s["model"] + s["new"].map({True: " **(new)**", False: ""}),
        "In-reg F1m": s["in_region_f1m"].map("{:.2f}".format),
        "Δ": s["delta"].map("{:+.2f}".format),
        "OOR F1m": s["oor_f1m"].map("{:.3f}".format),
        "95% CI": s.apply(lambda r: f"[{r.ci_lo:.2f}, {r.ci_hi:.2f}]", axis=1),
        "κ": s["kappa"].map("{:.2f}".format),
        "Xent": s["xent"].map("{:.2f}".format),
        "Seed SD": col("seed_f1m_sd").map(lambda v: num(v, "{:.3f}")),
        "Train / seed": col("per_seed_train_seconds").map(fmt_time),
    })
    parts += ["### Holdout and in-region scores",
              "Reference rows: MiniRocket from `../`, others from `out_of_sample/` (provenance-checked).", "",
              md(t)]

    sel = s[s["new"]].copy()
    if len(sel):
        st = pd.DataFrame({
            "Model": sel["model"],
            "Non-zero features / head / sample": col("mean_nonzero_per_head").loc[sel.index].map(lambda v: num(v, "{:.1f}")),
            "Effective features / head (exp entropy)": col("mean_effective_features_per_head").loc[sel.index].map(lambda v: num(v, "{:.1f}")),
            "Union of features selected": col("union_features_selected").loc[sel.index].map(lambda v: num(v, "{:.0f}")),
            "L1 non-zero features": col("l1_nonzero_features").loc[sel.index].map(lambda v: num(v, "{:.0f}")),
            "L1 chosen C (per seed)": col("l1_chosen_C").loc[sel.index].fillna(""),
        })
        parts += ["", "### How sparse is each head?",
                  "Gate statistics on validation samples (mean over seeds); L1 = features with any non-zero weight. "
                  "MiniRocket has 9,996 features.", "", md(st)]

    p = pd.read_csv(os.path.join(RES, "paired_bootstrap.csv"))
    if len(p):
        rows = []
        for (a, b), g in p.groupby(["A", "B"], sort=False):
            g = g.set_index("metric")
            f = lambda m: f"{g.loc[m,'delta']:+.3f} [{g.loc[m,'ci_lo']:+.3f}, {g.loc[m,'ci_hi']:+.3f}]"
            rows.append({"A": a, "B": b, "Question": g["question"].iloc[0], "Δ F1m [95% CI]": f("macro_f1"),
                         "Δ κ [95% CI]": f("kappa"), "McNemar p": f"{g['mcnemar_p'].iloc[0]:.3g}"})
        parts += ["", "### Paired field bootstrap (B = 10,000, seed 42)", "", md(pd.DataFrame(rows))]

    pc_dir = os.path.join(RES, "per_class")
    if os.path.isdir(pc_dir):
        pc = pd.DataFrame({f[:-4]: pd.read_csv(os.path.join(pc_dir, f), index_col=0)["f1-score"]
                           for f in sorted(os.listdir(pc_dir))})
        pc = pc.loc[[i for i in pc.index if i not in ("accuracy", "macro avg", "weighted avg")]]
        parts += ["", "### Per-class holdout F1", "", md(pc.map("{:.2f}".format).reset_index().rename(columns={"index": "Class"}))]

    for name, title in [("l1_equivalence.csv", "L1 solver check (FISTA vs scikit-learn saga, field, seed 42)")]:
        path = os.path.join(RES, name)
        if os.path.exists(path):
            parts += ["", f"### {title}", "", md(pd.read_csv(path).round(6))]
    for name, title in [("reproducibility_check.txt", "Reproducibility check"),
                        ("guard_check.txt", "Published and baseline files unchanged")]:
        path = os.path.join(RES, name)
        if os.path.exists(path):
            parts += ["", f"### {title}", "", "```", open(path).read().strip(), "```"]
    return "\n".join(parts)


def main():
    block = f"{BEGIN}\n{generated()}\n{END}"
    if os.path.exists(OUT):
        text = open(OUT).read()
        text = text[:text.index(BEGIN)] + block + text[text.index(END) + len(END):]
    else:
        text = ("# MiniRocket with L-TAE-S-style sparsity: results\n\n"
                "Design, hypotheses and disclosures: see README.md.\n\n## Results\n\n" + block +
                "\n\n## Reading\n\n_To be written after all runs finish._\n")
    open(OUT, "w").write(text)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
