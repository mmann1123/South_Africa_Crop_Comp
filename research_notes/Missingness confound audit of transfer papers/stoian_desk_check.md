# Stoian et al. 2019 desk check (Remote Sensing 11:1986)

Claim under test (audit card, `reports/Missingness confound audit of transfer papers.md`, line 57 and paragraph at line 181): in the leave-one-tile-out experiment over eleven French Sentinel-2 tiles (2016), the largest kappa drops fall on the mountain tiles (T30TYN, T31TGK), which the authors attribute to "tile content", while by the authors' own statement clear acquisitions run from about five in "northern and mountainous areas" to more than 30 in the Mediterranean, and iota2 linearly gap-fills everything to 33 dates. So the mountain drop could be a missingness (interpolation share) shift rather than a content shift.

Sources read: the full paper (HAL deposit hal-02627773, the published MDPI version, 26 pp. plus HAL cover; MDPI's own PDF endpoint returned "Access Denied" to curl), Figure 1 (tile map with eco-climatic zones, p. 5), Table 1 (class areas per tile, p. 6), Section 5.1 and Figure 6 (p. 14 to 15), Section 5.3 and Figure 19 (p. 22). Figure 19 was extracted as the embedded raster (739 x 312 px) and digitised by marker colour with a y-axis calibration from the tick labels. The Zenodo record 1474224 holds only the source code (`RT_DL_OSO_public-1.0.zip`, 66 kB), no data and no date lists. The Theia catalogue (theia.cnes.fr resto2) did not respond from this machine, so per-tile clear counts were measured independently from two public archives (see Section 2). No downloaded code was executed.

**Correction (29 Oct 2026, from the reproducible rebuild in `experiments/external_audits/stoian/`):** the T30TYN French-part median is 15 clear dates (10 for March to October), not 23 (17). The French mask is geoBoundaries FRA ADM0 at a pinned commit. The French-part correlation is unchanged (rho -0.58), and the mountain-tile verdict below stands. T31TGK's median longest March to October gap is 33 days, not 30.

## Verdict

**Weakens** the hypothesis as stated, while leaving a weaker, different version of it open.

The mountain-specific mechanism does not hold. The paper never gives per-tile clear counts, and the "five to more than 30" sentence is a France-wide range: none of the eleven tiles is in the Mediterranean zone (Figure 1). Measured independently for 2016, the two mountain tiles are not at the sparse end. With Sen2Cor scene classification (per-pixel, land pixels), T31TGK has a median of 23 clear dates (7th of 11, counting from the sparsest) and T30TYN 18 (tied 3rd; 23 on its French half). With L1C granule cloud cover, they are the two clearest tiles. T31TGK, the largest FG-UNET drop, sits in the upper half by clear dates and has the shortest growing-season gap of any tile (30 days). The sparsest tile, T31TDN (Centre, median 13), has only a mid-size FG-UNET drop and almost no RF drop (0.02).

A general density effect is still visible. Across all eleven tiles, the per-pixel clear count correlates negatively with the FG-UNET drop (Spearman rho = -0.60 for the full year and -0.65 for March to October, p about 0.03 to 0.05). It also correlates negatively with the RF drop, though not significantly (rho -0.40 to -0.48). The association gets stronger when the two mountain tiles are removed (rho -0.74, n = 9), so it is carried by lowland tiles, mainly T30TXQ (sparse, large drop) against T31TDJ, T31UDS and T30UXV (dense, small drop). It holds after partialling out a class-composition dissimilarity index (partial rho -0.60 for FG-UNET). Pearson correlations are weaker (r about -0.46 to -0.51, not significant) because T31TGK is a high-clear, high-drop outlier. With n = 11, a proxy cloud mask (Sen2Cor rather than the MAJA masks iota2 used) and kappas read off a figure, this is suggestive and not evidence.

The authors' own explanations fit the two extreme tiles better than missingness does. Table 1 shows that T30TYN and T31TGK hold almost all of the bare rock (BRO, 74,150 of 74,248 ha), all of the glacier (GPS, 2,582 ha) and 86% of the natural grassland (NGL). So when either is held out, the training pool contains almost no examples of its dominant classes. T30TXQ (the largest RF drop) has 85 ha of annual winter crops against 99,302 ha in total, which matches the authors' stated reason.

The card's framing, "the tiles with the largest drop are the tiles at the sparse end of the authors' own range", should be withdrawn. A defensible restatement is that per-tile acquisition density was not reported and does co-vary with the FG-UNET transfer drop across the lowland tiles. The mountain drops, however, coincide with class novelty and not with sparse sampling.

## 1. What the paper says

Section 2.1, p. 5: "linear temporal gapfilling is used to estimate image reflectance for pixels flagged as clouds, cloud shadows, saturation or non acquired and a subsequent temporal resampling is applied in order to reconstruct temporal profiles having identical time stamps ... Of course, the number of acquisitions for which the land is visible varies across tiles and even within every given tile. It ranges from five clear acquisitions in northern and mountainous areas to more than 30 in the sunny mediterranean areas where two satellite orbits overlap." Then: "each pixel at 10 m resolution is described by 330 features (10 spectral bands for 33 dates)."

Section 2.3, p. 6: "BRO is a rare class in most tiles except for T30TYN and T31TGK where it is one of the most represented classes."

Section 3 (patch sampling), p. 10: "mountain tiles do not contain beaches and vineyards while tiles covering plains do not contain glaciers. Moreover, coastal tiles contain large amounts of only one class: water."

Section 5.3, p. 22, Figure 19 ("Cohen's kappa comparison for all tiles between test of a model learned on 11 tiles and test on a 11th tiles of a model learned on 10 tiles"): "a drop in performance when a tile is not part of learning. Yet, it highly depends on the tile content. This drop is particularly high for mountain tiles (T30TYN and T31TGK) whose content is different from other tiles, and for T30TXQ which also has very few AWC labels. On the other hand, their is almost no drop for tile T30UXV." And: "FG-UNET suffers from a higher decrease than RF."

The paper gives no per-tile clear-date count, valid-pixel percentage, or map of clear observations, and no per-tile acquisition date list. Figure 1 is a map of tiles over eco-climatic zones, not of valid dates. Only one 2016 year is used, with Sentinel-2A alone (2B launched in 2017).

## 2. Per-tile clear acquisitions (measured here, not from the paper)

Two independent proxies for 2016 were used:

(a) **Per-pixel Sen2Cor SCL.** All 2016 L2A scenes for each tile were taken from the Microsoft Planetary Computer `sentinel-2-l2a` collection (63 to 97 unique dates per tile). SCL was read at about 320 m (a 344 x 344 grid per tile) and same-date granules were merged. "Clear" is SCL in {2, 4, 5, 6, 7, 11}, which keeps snow as valid. My understanding is that iota2 gap-fills only clouds, shadows, saturation and no-data from the MAJA masks, not snow, but I have not verified this against the 2016 iota2 version. Medians are over land pixels (sea excluded). The growing-season gap is the median per-pixel longest interval without a clear date between 1 March and 31 October.

(b) **L1C granule cloud cover.** From the Copernicus Data Space OData catalogue (N0500 reprocessing): the sum over granules of (1 - cloudCover/100) times the footprint's share of the tile. This is a tile-mean expected clear count.

Both are proxies for the Theia MAJA masks the authors used. Sen2Cor and the L1C mask both behave differently over snow and bright rock, which matters most for the two mountain tiles. The labelled pixels are not uniformly distributed, and the reference polygons were not available to restrict the counts to them. The absolute numbers (13 to 28 per pixel per year) are compatible with the authors' range of "five" to "more than 30". The 10th percentiles reach 8 to 9 in T30TYN and T30TWT.

## 3. Per-tile table

Kappa was read from Figure 19 (y-axis 300 px per kappa unit, so 1 px = 0.0033). Fully visible markers are accurate to about ±0.005. Markers partly hidden under another marker (marked *) are accurate to about ±0.015, and all four markers at T30UXV overlap. Drop = 11-tile kappa minus 10-tile (held-out) kappa for the same model. Positive means worse when the tile is held out.

| Tile | Region (Fig. 1 zone, centre) | SCL median clear dates, year (Mar to Oct) | SCL p10 | Gap Mar to Oct (d) | L1C eff. clear | FG-UNET 11 | FG-UNET 10+1 | FG drop | RF 11 | RF 10+1 | RF drop |
|---|---|---|---|---|---|---|---|---|---|---|---|
| T30TWT | Oceanic coast, Loire estuary | 22 (18) | 9 | 47 | 17.1 | 0.848 | 0.788 | 0.060 | 0.82* | 0.800 | 0.02 |
| T30TXQ | Oceanic, Landes forest and Gironde coast | 17 (14) | 14 | 50 | 19.1 | 0.913 | 0.655 | 0.258 | 0.855 | 0.687 | 0.168 |
| T30TYN | Mountain, central Pyrenees (about half in Spain) | 18 (13); French half 23 (17) | 9 (FR 8) | 48.5 | 26.8 | 0.610 | 0.465 | 0.145 | 0.668 | 0.587 | 0.081 |
| T30UXV | Oceanic, Normandy (Caen) | 24 (18) | 12 | 40 | 23.3 | 0.93* | 0.94* | -0.01 | 0.92* | 0.917 | 0.00 |
| T31TDJ | South-western basin, Tarn/Albi | 28 (22) | 18 | 30 | 25.8 | 0.803 | 0.777* | 0.026 | 0.765* | 0.730 | 0.035 |
| T31TDN | Degraded oceanic, Centre (Bourges) | 13 (9) | 10 | 50 | 16.6 | 0.913 | 0.785 | 0.128 | 0.88* | 0.857 | 0.023 |
| T31TEL | Mountain/semi-continental, Massif Central | 18 (15) | 13 | 40 | 19.9 | 0.850 | 0.708 | 0.142 | 0.792 | 0.730 | 0.062 |
| T31TGK | Mountain/altered Mediterranean, southern Alps | 23 (16) | 16 | 30 | 27.8 | 0.690 | 0.388 | 0.302 | 0.675 | 0.557 | 0.118 |
| T31UDQ | Degraded oceanic, Paris | 21 (16) | 10 | 50 | 20.2 | 0.840 | 0.685 | 0.155 | 0.713 | 0.650 | 0.063 |
| T31UDS | Oceanic coast, Calais/Dunkirk | 26 (21) | 22 | 50 | 26.2 | 0.922 | 0.892 | 0.030 | 0.86* | 0.840 | 0.02 |
| T32ULU | Semi-continental, Alsace/Vosges | 26 (20) | 16 | 33 | 22.7 | 0.902 | 0.788 | 0.114 | 0.863 | 0.820 | 0.043 |

Drop ranks. FG-UNET: T31TGK > T30TXQ > T31UDQ ≈ T30TYN ≈ T31TEL > T31TDN > T32ULU, then four tiles under 0.06. RF: T30TXQ > T31TGK > T30TYN, then the rest at or below 0.063. The two mountain tiles are among the three largest drops for both models, as the authors say. However, T30TXQ, a lowland forest and coast tile, has the largest RF drop and the second-largest FG-UNET drop.

## 4. Correlations (n = 11 unless stated)

| Clear-date measure | vs FG drop rho (p) / r | vs RF drop rho (p) / r | vs FG 10+1 kappa rho | vs RF 10+1 kappa rho |
|---|---|---|---|---|
| SCL median, year | -0.60 (0.05) / -0.46 | -0.40 (0.23) / -0.33 | +0.45 | +0.20 |
| SCL median, year, T30TYN French half | -0.58 (0.06) / -0.44 | -0.32 (0.34) / -0.29 | +0.36 | +0.14 |
| SCL median, Mar to Oct | -0.65 (0.03) / -0.51 | -0.48 (0.14) / -0.33 | +0.55 | +0.29 |
| SCL 10th percentile | -0.18 / -0.12 | +0.05 / 0.00 | +0.14 | +0.09 |
| Longest Mar to Oct gap (sign flipped) | -0.19 / +0.03 | -0.02 / -0.02 | -0.05 | -0.14 |
| L1C effective clear, year | +0.02 / 0.00 | +0.21 / +0.08 | -0.24 | -0.38 |
| L1C effective clear, Mar to Oct | -0.25 / -0.18 | +0.05 / +0.01 | -0.04 | -0.17 |

Without T30TYN and T31TGK (n = 9), the SCL median vs FG drop gives rho -0.74 (p 0.02), r -0.66, and vs RF drop rho -0.36. Leave-one-out on the 11-tile Spearman (SCL Mar to Oct vs FG drop) gives rho between -0.58 and -0.72 whichever tile is dropped. The two clear-date proxies disagree: L1C shows nothing, and it ranks the mountain tiles as the clearest, which is plausible because L1C scene cloud percentages behave differently over high terrain and it is a tile mean. The per-pixel SCL measure is closer to what iota2 interpolates. Treat all of these as n = 11 rank statistics with roughly ±0.3 sampling noise on rho and no correction for testing about 28 pairs.

**Class composition as the alternative.** I computed, for each tile, the Jensen-Shannon divergence between its Table 1 class distribution (water excluded) and the pooled distribution of the other ten tiles. It correlates with the FG drop at rho +0.49 (p 0.13) and with the RF drop at +0.32. That is comparable to, not better than, clear dates. Clear dates and composition divergence are only weakly related (rho -0.31). Controlling each for the other: clear dates | composition gives partial rho -0.60 (FG) and -0.44 (RF); composition | clear dates gives +0.40 (FG) and +0.18 (RF). A two-variable OLS explains little (R² 0.30 for FG, 0.15 for RF). The share of mountain classes (BRO + NGL + GPS) explains the absolute held-out kappa (Pearson r -0.85 FG, -0.75 RF). That correlation is driven by the two mountain tiles, and those tiles are also the worst in-sample (FG-UNET 11-tile kappa 0.61 and 0.69), so their difficulty is not only a transfer effect.

## 5. Adversarial points

Several tiles with few clear dates transfer reasonably well. T31TDN is the sparsest tile (median 13, 9 in-season, 50-day gaps), yet it loses only 0.023 RF kappa and 0.128 FG-UNET. T30TWT shares the lowest p10 (9) with T30TYN and loses 0.06 and 0.02.

Some tiles with many clear dates transfer badly. T31TGK (median 23, shortest gap) has the largest FG-UNET drop. T32ULU (26) loses 0.114 FG-UNET.

The authors' content explanations are concrete and checkable in Table 1. The mountain tiles hold almost all BRO, all GPS and 86% of NGL (69,902 of 80,987 ha), and T30TXQ has almost no AWC. Their explanation for T30TXQ fits the RF result best.

The premise sentence ("five ... in northern and mountainous areas to more than 30 in the sunny mediterranean areas where two satellite orbits overlap") does not map onto these tiles. None is Mediterranean (Figure 1), and the measured per-tile medians span only 13 to 28.

Half of T30TYN is in Spain, where it is drier and clearer. Its French half (median 23) is no sparser than the tile as a whole, so this does not rescue the mountain mechanism.

## 6. Not accessible or not done

Per-tile clear-date counts, Theia L2A date lists, and the MAJA cloud masks actually used are not in the paper, the code repository or the Zenodo record (code only). The Theia catalogue API timed out from this machine. The reference polygons were not obtained, so clear counts are over all land pixels and not the labelled pixels. Figure 19 has no error bars, and the leave-one-tile-out experiment was run once, while the in-sample runs were repeated five times for Figure 7 only. The 2016 Sen2Cor SCL on Planetary Computer comes from a later reprocessing and differs from the MAJA masks of 2016 iota2. The snow-handling assumption for iota2 gap-filling was not verified against the iota2 version used.

Scratch artefacts (not part of the repo): `/tmp/claude-1000/-home-mmann1123-Documents-github-South-Africa-Crop-Comp/6a949ad2-1d7c-45e4-b537-97b8e63decc6/scratchpad/stoian_read/` (paper PDF `hal.pdf`, Figure 19 raster `img/f19-000.png`, per-tile SCL stacks `scl_*.npz`, summaries `scl_summary.json`, `tile_clear.json`, granule list `cdse2016.jsonl`). The scripts are in the sibling `scripts/` directory.
