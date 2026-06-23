# From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification

**Citation:** Martins, A. F. T. & Astudillo, R. F. (2016). *From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification.* Proceedings of the 33rd International Conference on Machine Learning (ICML 2016), PMLR v48, pp. 1614–1623. arXiv:1602.02068. DOI: not found in PDF (PMLR proceedings paper; no Crossref DOI assigned — identify by PMLR v48 / arXiv:1602.02068).
**BibTeX key:** `martins2016softmax`

> **Note:** this is the foundational **sparsemax** paper — *not* the Lightweight Temporal Attention Encoder (Garnot et al.; see `ltae2020_Garnot_LTAE.pdf`), under which name it had originally been misfiled. It is relevant to the L-TAE line of work only because L-TAE-style temporal attention is built on softmax, and sparsemax/entmax are the sparse-attention alternatives.

## Objectives

This is a machine-learning theory/methods paper, not a remote-sensing application (authoring rule 7 applies — there is no spatial evaluation protocol to assess). It proposes **sparsemax**, an alternative to the softmax transformation that can output genuinely sparse probability distributions (assigning exact zero probability to some outputs), and develops the supporting theory: a closed-form evaluation, an efficient Jacobian for backpropagation, and a convex "sparsemax loss" that is the sparse analogue of logistic/cross-entropy loss. The motivation is twofold: (1) multi-label classification, where a model should select a subset of labels, and (2) **attention mechanisms**, where a sparse attention distribution focuses on a small, interpretable set of inputs while zeroing out the rest.

## Methods

- **Sparsemax definition.** `sparsemax(z) = argmin_{p ∈ Δ}‖p − z‖²` — the Euclidean projection of the score vector `z` onto the probability simplex. Unlike softmax (which always has full support: every output is strictly positive), this projection frequently lands on a face of the simplex, producing exact zeros.
- **Closed form and threshold.** Sparsemax is computed by sorting `z`, finding a support set, and subtracting a threshold `τ(z)`: `p_i = [z_i − τ(z)]_+`. Coordinates below the threshold are clipped to zero. In the binary case sparsemax reduces to a **hard sigmoid**.
- **Jacobian.** The paper derives the sparsemax Jacobian and shows it is *cheaper* than softmax's (it only involves the support set), enabling faster gradient backprop.
- **Sparsemax loss.** A new convex, everywhere-differentiable loss whose gradient mirrors the logistic-loss gradient (prediction minus target); shown to be a multi-class generalization of the **Huber classification loss**.
- **Experiments.** (1) Multi-label classification on benchmark datasets (linear classifiers) comparing sparsemax loss to logistic and softmax baselines; (2) a **neural selective-attention** mechanism using sparsemax, applied to a natural-language-inference task, compared against softmax attention.

**Evaluation protocol:** Standard ML benchmark splits on text/NLI datasets — not applicable to the spatial-transfer spectrum our manuscript is organized around. No imagery, no geographic generalization, no spatial leakage concern. The relevant takeaway is methodological, not empirical-comparator.

## Key Findings

- **Sparsemax produces sparse, selective distributions** while retaining softmax's convenient properties: simple to evaluate, differentiable, convex companion loss. It is "even cheaper to differentiate" than softmax.
- **On multi-label classification**, sparsemax loss is competitive with the best baselines, with the advantage of directly predicting a *set* of labels (sparse support) rather than thresholding a dense softmax.
- **On attention-based NLI**, sparsemax attention matches softmax performance but yields a **"selective, more compact attention focus"** — the model attends to a small, human-readable subset of tokens, improving interpretability without accuracy cost.
- **Theoretical unification**: the sparsemax loss connects to the Huber loss, situating it within robust-statistics M-estimation.

## Relevance to Our Crop-Classification Study

The link is conceptual — about inductive bias and interpretability — not a direct empirical comparator. It bears on the manuscript's argument in three ways:

1. **Sparse attention as a transfer-friendly inductive bias.** Our manuscript's central narrative is that *sparse, feature-selecting* models (gradient-boosted trees, TabNet) transfer to a spatially disjoint holdout better than *dense* temporal/patch nets. Sparsemax is the attention-mechanism instantiation of exactly that bias: it forces the model to commit to a small support and ignore the rest. Where our L-TAE uses softmax temporal attention (every acquisition date receives nonzero weight), a sparsemax/entmax variant would attend to only the phenologically informative dates and zero out the rest — plausibly a more transferable temporal representation. This is a concrete, citable mechanism for *why* sparsity aids domain transfer, and a natural "future work" lever for our L-TAE.

2. **Interpretability of temporal attention.** A recurring selling point of temporal-attention models in crop classification is that attention weights reveal *which dates* drive a class decision (e.g., a crop's green-up or senescence window). Softmax attention dilutes this signal across all dates; sparsemax gives a crisp, sparse set of "decision dates." If we discuss interpretability of our L-TAE, this paper is the principled basis for sharpening it.

3. **TabNet connection.** TabNet — one of our best-transferring models — uses **sparsemax** explicitly in its sequential attentive feature-selection masks. Citing Martins & Astudillo (2016) documents the origin of TabNet's sparse feature-selection mechanism and reinforces our framing that TabNet's transfer advantage stems from a learned, sparse, tree-like feature-selection inductive bias rather than dense representation learning. This is arguably the single most direct relevance: the sparsemax operator is *inside* a model we benchmark.

## Evaluation Caveats

- **Not a remote-sensing or crop paper.** No imagery, no spatial holdout, no class-imbalance/minority-recall analysis, no macro-F1/Kappa — none of the empirical-comparator dimensions our other briefers track. It cannot be cited for any accuracy comparison.
- **No domain-transfer evidence.** The paper demonstrates sparsity and interpretability on NLI/multi-label text benchmarks; it makes *no* claim about out-of-distribution or geographic generalization. The transfer-friendliness argument (relevance point 1) is our extrapolation from the inductive bias, not something the paper tests.
- **Sparsity ≠ better accuracy.** The NLI result is that sparsemax *matches* softmax, not that it beats it. The benefit is compactness/interpretability, so any claim we make should be framed as "comparable accuracy with sparser, more interpretable, potentially more transferable attention," not "higher accuracy."
- **Superseded in places.** Subsequent work (entmax / α-entmax, Peters et al. 2019) generalizes sparsemax with a tunable sparsity parameter and is what most modern sparse-attention implementations use. If we pursue a sparse-attention L-TAE variant, entmax is the more current operator to cite alongside this foundational paper.
- **Filename mismatch is a citation hazard.** Anyone wiring up the bibliography from the filename alone could miscite this as an L-TAE paper. The correct cite key is the sparsemax/Martins-Astudillo-2016 entry; the L-TAE paper is the separate Garnot et al. file.
