# Mapping crops within the growing season across the United States

**Citation:** Venkata Shashank Konduri, Jitendra Kumar, William W. Hargrove, Forrest M. Hoffman, and
Auroop R. Ganguly (2020). "Mapping crops within the growing season across the United States."
*Remote Sensing of Environment*, 251, 112048. DOI: `10.1016/j.rse.2020.112048` (extracted from the
PDF first page; verified against Crossref on 2026-11-09: the returned title matches exactly). Open
access (CC BY-NC-ND). Source for this briefer: the full text of the PDF in this folder (the main
article; the Supplementary Material with Tables S1-S4 and Figs. S1-S5 was not available, so
per-crop and per-year numbers that live only there are not reproduced).

## Objectives

The paper maps crops across the continental United States (CONUS) while the season is still running,
and estimates the earliest date at which each crop can be mapped at 90 percent of its full-season
accuracy. The method is an unsupervised-then-supervised "cluster-then-label" scheme on 8-day
MODIS NDVI, with the USDA Cropland Data Layer (CDL) as the only label source. A secondary goal is
to back-cast annual CDL-style maps for 2000-2007, before the CDL existed. The target is eight
area-dominant crops (corn, soybeans, winter wheat, fallow/idle cropland, other hay/non alfalfa,
alfalfa, sorghum, rice), although about 100 CDL classes are mapped (Table S2).

## Methods

### Data and clustering

NDVI comes from the ORNL DAAC "smoothed and gap-filled" MODIS product built from Terra MOD13Q1 and
Aqua MYD13Q1 Collection 5 at 231 m and 8 days, processed with the NASA Stennis Time Series Product
Tool (TSPT). The authors say this tool is used "to remove clouds and otherwise clean and filter the
time series temporally" and that the result "is nearly complete, with few missing values" (p. 3).
No QA-flag handling, no valid-observation counts, and no filter parameters are described in the
paper itself. Multivariate Spatio-Temporal Clustering (MSTC, a non-hierarchical k-means) groups each
annual NDVI trajectory, over the full 2000-2018 series, into 5,000 "phenoregions". Phenoregions are
regridded to 30 m (nearest neighbour) to match the CDL. Separately, 500 climatic "ecoregions" are
built from 15 bioclimatic, topographic and soil variables (Table S1).

### Labelling and within-season mapping

Each phenoregion receives one crop label by majority spatial overlap with the CDL, scored by the
Mapcurves goodness-of-fit statistic, independently within each ecoregion and pooled over the
training years 2008-2014. Within-season classification assigns a partial-year NDVI trajectory to the
closest phenoregion, then applies the stored label. Output is discarded until a pixel passes a
greenness threshold of 20 percent of the phenoregion's annual NDVI amplitude (Fig. 3). Models are
compared at CONUS, state and ecoregion scale; ecoregion scale wins, with diminishing returns for
major crops.

### Evaluation protocol

**Evaluation protocol:** a mixed design, and the cross-year part is the only transfer. The model is
"trained and validated on the years 2008-2014, then tested independently on 2015-2018" (abstract).
Within 2008-2014, the CDL pixels of each year inside each ecoregion are split randomly 70/30 into
training and validation (Section 3.4.1), a pooled random pixel split with no field or spatial block
disjointness, so the 60-61 percent validation figure is an in-region, in-year number with the usual
spatial-autocorrelation optimism (although the label is assigned to whole phenoregions, which
softens but does not remove leakage). The 2015-2018 test is cross-year, same geography: the
ecoregions and phenoregions are identical between training and test, and the phenoregions were
clustered on the full 2000-2018 NDVI record, so the test years' NDVI contributed to the cluster
definition (the labels did not). No region is held out; the paper does not do spatial transfer.
Metrics are producer's, user's and overall accuracy, plus county and state acreage R-squared. No
macro-F1, Kappa, or cross-entropy is reported, and reference labels are the CDL itself, a
classification product, not ground truth.

## Key Findings

Training-period (2008-2014, 30 percent random validation) overall accuracy is "60-61%" over 102
crop types. User's accuracy for corn, soybeans and winter wheat is 61-67, 58-65 and 60-72 percent;
for the other five major crops it is 50-60 percent (p. 6). On the unseen years 2015-2018, overall
accuracy over all 102 types is "~58%" for 2015-2017 and "53%" for 2018. User's accuracy for corn,
soybeans and winter wheat is "~60%" except corn in 2018, which drops to 54 percent, and soybean
producer's accuracy in 2018 falls to 39 percent (p. 6). The transfer loss is therefore small (about
2-3 points overall) except in 2018.

Ecoregion-level pixel overall accuracy for 2015 is about 70 percent across much of the Corn Belt,
above 85 percent in parts of Kansas, Oklahoma and Texas, about 60 percent in diverse Dakota,
Mississippi and Wisconsin ecoregions, and about 50 percent in the California Central Valley. Accuracy
tracks Shannon crop diversity (Fig. 8) and field size relative to the 231 m pixel. County and state
acreage aggregates agree with the CDL far better than pixels (Fig. 9), which the authors attribute
partly to pixel errors cancelling on aggregation.

Within-season, winter wheat, corn and soybeans reach 90 percent of full-season accuracy by early,
mid and late August in 2015, other hay and fallow by mid-August, alfalfa by late September, and
sorghum and rice only around mid-November (p. 8-9). The earliest dates vary spatially (corn by
mid-May in southern Minnesota against late October in eastern Indiana) and modestly between years.

### How the paper explains degradation

The explanation is phenological and meteorological only. "Shifts in crop phenology from year to
year were the major source of variability in crop mapping accuracy" (p. 11). For 2018, corn user's
accuracy in Iowa, Illinois, Indiana and Nebraska falls "from 69% to 57%" and soybean producer's
accuracy "from 58% to 27%" (Table 2), attributed to "Unusually early planting of soybeans, coupled
with unusually fast phenological development" (Fig. S5), which made soybean NDVI resemble corn.
Other stated limits are the coarse 231 m pixel and mixed pixels for small fields, CDL label error,
and the winner-takes-all phenoregion label. The paper does not test missing data, cloud cover or
product quality as a cause of the 2018 drop or any other loss.

## Relevance to Our Crop-Classification Study

The paper is a clean cross-year transfer example at national scale with a modest, honestly reported
generalization gap, and it is a useful counterweight to papers that report only in-year accuracy:
the authors stress that prior work was "tested on the same years as they were developed" and
that realistic use means mapping an unseen year. It also shows temporal (year) transfer failing for
a phenological reason, which parallels our spatial-holdout story where in-region k-fold hides
drift. Architecturally it sits at the sparse, low-capacity end: a 5,000-prototype lookup table, no
neural network, with spatial specialization (ecoregions) giving most of the gain, in line with our
finding that constrained models transfer well.

For the missingness question that motivates our audit, the paper is a data-source case rather than a
gap-handling case. Cloud and snow gaps were repaired upstream by an external smoothing and
gap-filling tool, applied to all years by the same process, and the paper reports no statistic on
how much of the NDVI series was filled in any year. The cross-year test is therefore a transfer
loss with a described-but-unquantified, domain-identical gap treatment, and the loss is explained
by weather-driven phenology without a missingness check. Because the gap-filler's reliance is
unmeasured, a year with more cloud or snow would carry more interpolated NDVI without the paper
being able to say so.

Not directly comparable to our numbers: overall pixel accuracy against CDL labels, with pooled random
validation, over 102 classes.

## Evaluation Caveats

Pooled random 70/30 pixel split within ecoregion and year (spatial leakage in the validation
figure). The test years share geography with training and their NDVI shaped the 5,000 phenoregions,
so the held-out test is not fully independent. Labels are the CDL, itself a model product with
self-reported 85-95 percent accuracy for major crops. Overall accuracy is dominated by area-weighted
major crops, with minority-crop producer's accuracy of only 30-40 percent for sorghum and rice;
no balanced metric. The within-season design needs the 20 percent greenness gate, so early-season
accuracy is computed only on pixels that pass it. Per-year data-quality or valid-observation
statistics are not reported anywhere in the main text. Spatial transfer, other sensors, and
computational cost are not measured (the authors do note scalability as a design goal).
