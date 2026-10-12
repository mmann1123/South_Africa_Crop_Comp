"""Regenerate the tables in RESULTS.md from results/*.csv.

Only the block between the GENERATED markers is replaced; hand-written sections
around it are kept. Creates RESULTS.md with an empty skeleton if it does not exist.

  ~/miniconda3/envs/ts_baselines/bin/python experiments/recent_baselines/make_report.py
"""
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "RESULTS.md")
BEGIN, END = "<!-- BEGIN GENERATED (make_report.py) -->", "<!-- END GENERATED -->"


def md(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r.values) + " |")
    return "\n".join(lines)


def fmt_time(s):
    if pd.isna(s):
        return ""
    return f"{s:.0f} s" if s < 120 else f"{s/60:.1f} min"


def generated():
    parts = []
    s = pd.read_csv(os.path.join(RES, "summary.csv"))
    s = s.sort_values(["new", "oor_f1m"], ascending=[False, False])
    t = pd.DataFrame({
        "Model": s["model"] + s["new"].map({True: " **(new)**", False: ""}),
        "In-reg F1m": s["in_region_f1m"].map("{:.2f}".format),
        "Δ": s["delta"].map("{:+.2f}".format),
        "OOR F1m": s["oor_f1m"].map("{:.2f}".format),
        "95% CI": s.apply(lambda r: f"[{r.ci_lo:.2f}, {r.ci_hi:.2f}]", axis=1),
        "κ": s["kappa"].map("{:.2f}".format),
        "wF1": s["f1_weighted"].map("{:.2f}".format),
        "Xent": s["xent"].map("{:.2f}".format),
        "Seed SD (OOR F1m)": s.get("seed_f1m_sd", pd.Series(dtype=float)).map(
            lambda v: "" if pd.isna(v) else f"{v:.3f}"),
        "Train time / seed": s.get("per_seed_train_seconds", pd.Series(dtype=float)).map(fmt_time),
    })
    parts += ["### Holdout and in-region scores",
              "New models are trained here; published rows are re-scored from `out_of_sample/` "
              "and match Table 1. In-region = internal 20% field split; Δ = OOR − in-region.", "",
              md(t)]

    p = pd.read_csv(os.path.join(RES, "paired_bootstrap.csv"))
    if len(p):
        w = p.pivot_table(index=["A", "B", "question"], columns="metric",
                          values=["delta", "ci_lo", "ci_hi"], aggfunc="first")
        mc = p.drop_duplicates(["A", "B"]).set_index(["A", "B"])["mcnemar_p"]
        rows = []
        for (a, b, q), r in w.iterrows():
            rows.append({"A": a, "B": b, "Question": q,
                         "Δ F1m [95% CI]": f"{r[('delta','macro_f1')]:+.3f} [{r[('ci_lo','macro_f1')]:+.3f}, {r[('ci_hi','macro_f1')]:+.3f}]",
                         "Δ κ [95% CI]": f"{r[('delta','kappa')]:+.3f} [{r[('ci_lo','kappa')]:+.3f}, {r[('ci_hi','kappa')]:+.3f}]",
                         "Δ acc [95% CI]": f"{r[('delta','accuracy')]:+.3f} [{r[('ci_lo','accuracy')]:+.3f}, {r[('ci_hi','accuracy')]:+.3f}]",
                         "McNemar p": f"{mc.loc[(a, b)]:.3g}"})
        order = {(a, b): i for i, (a, b, _) in enumerate(
            p.drop_duplicates(["A", "B"])[["A", "B", "question"]].itertuples(index=False))}
        rows.sort(key=lambda r: order[(r["A"], r["B"])])
        parts += ["", "### Paired field bootstrap (B = 10,000, seed 42)",
                  "Δ = A − B on the same resampled holdout fields; a CI excluding 0 is significant at 95%.", "",
                  md(pd.DataFrame(rows))]

    pc_dir = os.path.join(RES, "per_class")
    if os.path.isdir(pc_dir):
        tabs = {}
        for f in sorted(os.listdir(pc_dir)):
            d = pd.read_csv(os.path.join(pc_dir, f), index_col=0)
            tabs[f[:-4]] = d["f1-score"]
        pc = pd.DataFrame(tabs)
        pc = pc.loc[[i for i in pc.index if i not in ("accuracy", "macro avg", "weighted avg")]]
        pc = pc.map("{:.2f}".format).reset_index().rename(columns={"index": "Class"})
        parts += ["", "### Per-class holdout F1 (new models)", "", md(pc)]

    seed_rows = []
    for d in sorted(os.listdir(os.path.join(HERE, "models"))):
        mp = os.path.join(HERE, "models", d, "metadata.json")
        if not os.path.exists(mp) or any(t in d for t in ("smoke", "repro")):
            continue
        m = json.load(open(mp))
        for s in m["per_seed"]:
            seed_rows.append({"Model": d, "Seed": s["seed"], "Train time": fmt_time(s["train_seconds"]),
                              "Epochs / chosen C": s.get("epochs_run", s.get("chosen_C")),
                              "Best val F1m": s.get("best_val_f1_macro",
                                                    max(s.get("val_f1_by_C", [float("nan")])))})
    if seed_rows:
        parts += ["", "### Per-seed training", "", md(pd.DataFrame(seed_rows))]

    for name, title in [("reproducibility_check.txt", "Reproducibility check"),
                        ("guard_check.txt", "Published files unchanged")]:
        path = os.path.join(RES, name)
        if os.path.exists(path):
            parts += ["", f"### {title}", "", "```", open(path).read().strip(), "```"]
    return "\n".join(parts)


def main():
    block = f"{BEGIN}\n{generated()}\n{END}"
    if os.path.exists(OUT):
        text = open(OUT).read()
        if BEGIN in text and END in text:
            text = text[:text.index(BEGIN)] + block + text[text.index(END) + len(END):]
        else:
            text = text.rstrip() + "\n\n" + block + "\n"
    else:
        text = ("# R2-Q3a recent baselines: results\n\n"
                "Setup and fairness: see README.md.\n\n## Results\n\n" + block +
                "\n\n## Reading\n\n_To be written after all runs finish._\n")
    open(OUT, "w").write(text)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
