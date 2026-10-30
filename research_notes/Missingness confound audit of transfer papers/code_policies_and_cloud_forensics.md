# Code/data-availability policies of remote-sensing venues, and after-the-fact cloud/gap forensics for holdout regions

Researched 2026-10-28. Companion files written alongside this note (same folder):

- `earthsearch_harvest.py` — the script used to pull Sentinel-2 L1C granule-level `eo:cloud_cover` from AWS Earth Search for 33 cropland points (2018–2024).
- `earthsearch_s2_l1c_cloudcover_by_month_2018-2024.csv` — its output (per region × month: n granules, median cloud %, fraction of granules < 20 % and < 50 % cloudy).
- `earthsearch_s2_l1c_cloudcover_SA_tiles_2017.csv` — the same harvest run on the exact AI4FoodSecurity train/holdout label extents for calendar 2017.

Access caveat that shaped Part 1: ScienceDirect "Guide for Authors" pages (RSE, ISPRS JPRS, CEA, JAG, AFM, Science of Remote Sensing) and MDPI's instructions page sit behind bot checks and could not be read directly; Elsevier's publisher-level policy pages and third-party policy registries were used instead, and that is flagged per journal below.

---

## KQ1. Which venues (since roughly 2019–2021) make code access expected, so a paper's silence on missing-data handling can be checked against its repo?

### Takeaway
No remote-sensing journal in the list *requires* code deposit as of 2026; the only venues where custom code must be shared with editors/reviewers are Nature Communications (code-availability statement mandatory for custom code central to the conclusions, with code peer review) and Scientific Data (code availability mandatory, reviewers must be able to access it). Every Elsevier, IEEE, MDPI, Taylor & Francis, Copernicus and ML-conference venue examined either encourages code or is silent, so for the great majority of transfer papers the auditable artefact is the *data availability statement* (mandatory at MDPI and Springer Nature, journal-dependent at Elsevier, encouraged at IEEE), not the code.

### Cited Findings

**Elsevier journals (RSE, ISPRS JPRS, JAG, CEA, AFM, Science of Remote Sensing)**

- Elsevier does not mandate sharing publisher-wide: "we encourage and support authors to share their research data rather than mandating them to do so… some of our journals may reflect this with their own mandatory data sharing policies"; research data is defined to include "software and algorithms". — [Elsevier research data FAQ](https://www.elsevier.com/about/policies-and-standards/research-data/faq)
- Each Elsevier journal adopts one of five options. Option A: deposit "encouraged". Option B: deposit encouraged "or a Research Data Availability Statement… If sharing your research data isn't possible, please make a statement explaining why… The statement will appear in your published article". Option C: "You are required to: Deposit your research data in a relevant data repository… If sharing your research data isn't possible, make a statement explaining why". Option D: deposit required, no statement alternative. Option E: deposit required and "Peer reviewers are asked to review the data prior to publication". — [Elsevier research data guidelines](https://www.elsevier.com/researcher/author/tools-and-resources/research-data/data-guidelines)
- The Elsevier "data statement" is described as something "a number of Elsevier journals encourage their authors to" provide; it appears "under the 'research data' section from the article outline". No effective date is given. — [Elsevier data statement page](https://www.elsevier.com/researcher/author/tools-and-resources/research-data/data-statement)
- A secondary registry (CASRAI, updated 24 Jul 2026) states that "most Elsevier journals require a DAS as part of one of the Research Data options", with no dates. — [CASRAI publisher comparison](https://www.casrai.org/compare/data-sharing-policy-publisher-comparison)
- Remote Sensing of Environment is listed by CIRAD's "Où publier" registry (updated 11 Feb 2025) with "Research data access policy: Deposit recommended", pointing to Elsevier's recommended repositories; Science of Remote Sensing is also listed as "Deposit recommended". — [CIRAD entry for RSE](https://ou-publier.cirad.fr/en/node/3576); search result summary for Science of Remote Sensing from the same registry.
- A consultancy summary of RSE's guide lists a "Data availability statement with repository links for remote-sensing datasets, ground-truth measurements, or analysis code" among items required at submission (secondary source, not the journal's own text). — [Manusights RSE guide](https://manusights.com/blog/remote-sensing-of-environment-submission-guide)
- Elsevier's "Research Elements" are a suite of open-access, peer-reviewed journals (Data in Brief, MethodsX, SoftwareX, Software Impacts, HardwareX…) whose articles "link to the research object and associated original research articles". — [Elsevier Research Elements](https://www.elsevier.com/researcher/author/tools-and-resources/research-elements)
- Software Impacts (ISSN 2665-9638) publishes short "Original Software Publication" articles with open-source code; eligible software must already have been used in "at least one scholarly, peer-reviewed article", and "Certified papers will be given additional exposure by receiving a Reproducibility Badge, and by being listed on the CodeOcean website". — [Software Impacts journal page](https://shop.elsevier.com/journals/software-impacts/2665-9638)

**IEEE GRSS journals (TGRS, JSTARS, GRSL) and IGARSS**

- IEEE-wide policy: "All IEEE authors are encouraged to share their data, code, and other research outputs." Code Ocean: "Authors who have published with IEEE in the past five years can upload their code to Code Ocean"; "Articles with linked code have a Code & Datasets tab" in Xplore. Reviewer access to code is not addressed. — [IEEE Author Center, Research Reproducibility](https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/research-reproducibility/)
- The IEEE–Code Ocean partnership page is dated March 14, 2017 ("IEEE and Code Ocean recently partnered to enable IEEE authors to upload, share, and run their code"). — [IEEE Innovate](https://innovate.ieee.org/ieee-partners-with-code-ocean/)
- TGRS "Information for Authors" (PDF generated 1 Dec 2023) contains no code, data, reproducibility, Code Ocean or DataPort clause; its only relevant sentence is "Experimental data must be complete and include sufficient descriptions of the experimental apparatus, methods, and relevant experimental conditions." Review is single-blind. — [TGRS Information for Authors PDF](https://www.grss-ieee.org/wp-content/uploads/2023/12/Information-for-Authors_TGRS_pdf.pdf)
- JSTARS's GRSS page lists an "Optional Code Ocean Submission" link. — [GRSS JSTARS page](https://www.grss-ieee.org/publications/journal-of-selected-topics-in-applied-earth-observations-and-remote-sensing/)
- GRSL is listed by CIRAD with "Research data access policy: Deposit recommended". — [CIRAD entry for GRSL](https://ou-publier.cirad.fr/en/node/4962)
- The GRSS Remote Sensing Code Library (RSCL) is described in the past tense and "is not currently accepting new submissions". — [RSCL About](https://tools.grss-ieee.org/rscl1/about.html)
- IGARSS 2026 solicits "Full 4-page papers required for inclusion in IEEE Xplore" and abstracts; no code or data requirement appears in the call. — [IGARSS 2026 listing](https://eo.belspo.be/nl/agenda/igarss-2026)

**MDPI Remote Sensing**

- The journal's instructions list "Data Availability Statement" among mandatory back-matter sections, and its Earth Observation Data section states "a clear data availability statement is required for all submissions to this section" (search-result snippets; page itself blocked). — [MDPI Remote Sensing instructions](https://www.mdpi.com/journal/remotesensing/instructions); [Earth Observation Data section](https://www.mdpi.com/journal/remotesensing/sections/earth_observation_data)
- CASRAI: at MDPI the DAS is "required on every published article across MDPI journals" while "Deposit in a trusted repository is recommended, not strictly mandatory." No code requirement. — [CASRAI publisher comparison](https://www.casrai.org/compare/data-sharing-policy-publisher-comparison)

**Nature Communications (Nature Portfolio) and Scientific Data**

- "All published manuscripts reporting original research in Nature Portfolio journals must include a data availability statement." "Authors must make data available for purposes of peer review, if requested by reviewers". — [Nature Portfolio reporting standards](https://www.nature.com/nature-portfolio/editorial-policies/reporting-standards)
- Code: "For all studies using custom code or mathematical algorithm that is deemed central to the conclusions, a statement must be included under the heading 'Code availability'… Nature journals undertake peer review of custom code… when it is central to the manuscript. For code/algorithm peer review, we require sharing of code/algorithm during the peer review process with editors and reviewers"; editors "reserve the right to decline the paper if important code is unavailable"; Code Ocean is offered as an optional free service. The page cites the editorials "Code share" (Oct 2014) and "Does your code stand up to scrutiny?" (Mar 2018). — same page.
- Effective dates for the DAS: trial at five Nature journals including Nature Communications from March 2016; mandatory for Nature and 12 other titles from September 2016, "across the diverse range of Nature journals by early 2017". — [Nature 537, 138 (2016) "Where are the data?"](https://www.nature.com/articles/537138a)
- Springer Nature announced on 22 March 2023 that it is "requiring mandatory data availability statements (DAS) across its journals portfolio", rolled out "progressively". — [Springer Nature press release](https://group.springernature.com/us/group/media/press-releases/archive-2023/springer-nature-makes-data-sharing-easier-with-single-policy/25211176)
- Scientific Data: "Authors must make available, to editors, reviewers, and eventual readers, any previously unreported custom computer code or algorithm used to generate or process the data"; a "Code availability" statement is mandatory even if no custom code was used. — [Scientific Data policies](https://www.nature.com/sdata/policies/editorial-and-publishing-policies)

**Earth System Science Data (Copernicus)**

- "upon submission, all data must possess either a functional digital object identifier (DOI)" or review access; "upon publication, all data must possess a functional DOI and citation"; "authors must certify some form of fully anonymous review access, directly at the chosen repository"; a "Data availability" section is required. No code requirement is stated (code is merely named among research outputs). — [ESSD data policy](https://www.earth-system-science-data.net/policies/data_policy.html)

**GIScience & Remote Sensing (Taylor & Francis)**

- T&F tiers: Basic policy "encourages you to share… You are also encouraged to cite data and provide a data availability statement"; "Share upon reasonable request"; "Publicly Available: You must make your data freely available"; "Open Data: … under a license allowing re-use by any third party"; "Open and FAIR". No introduction dates on the page. — [T&F data sharing policies](https://authorservices.taylorandfrancis.com/understanding-our-data-sharing-policies/)
- Which tier GIScience & RS applies could not be confirmed from the journal page (blocked). A CIRAD registry entry (2024) reportedly lists the journal as requiring deposit; the journal became fully open access from Volume 59 (2022). — search summaries citing [CIRAD](https://ou-publier.cirad.fr/) and [T&F](https://www.tandfonline.com/action/journalInformation?show=aimsScope&journalCode=tgrs20); unverified.

**Computer-vision and ML venues**

- CVPR 2025 Author Suggested Practices: "We highly encourage authors to voluntarily submit their code as part of supplementary material… We expect (but do not require) that the accompanying code will be submitted with accepted papers. Reviewers may optionally check this code"; datasets claimed as contributions are "expected… publicly available no later than the camera-ready deadline". — [CVPR 2025](https://cvpr.thecvf.com/Conferences/2025/AuthorSuggestedPractices); CVPR 2024 "strongly encourage[s]" code and data release. — [CVPR 2024](https://cvpr.thecvf.com/Conferences/2024/AuthorSuggestedPractices)
- EarthVision (CVPR workshop, GRSS-sponsored): double-blind, CVPR formatting and plagiarism rules apply; no code or dataset release requirement appears in the 2021–2026 calls. — [EarthVision 2026](https://www.grss-ieee.org/events/earthvision-2026/); [EarthVision 2021 CFP](https://www.grss-ieee.org/?p=785)
- NeurIPS paper checklist (mandatory section since 2021; "The papers not including the checklist will be desk rejected"): asks "Did you include the code, data, and instructions needed to reproduce the main experimental results", but "no is an acceptable answer" and papers "cannot be rejected simply for not including code, unless this is central to the contribution". — [NeurIPS Paper Checklist guide](https://neurips.cc/public/guides/PaperChecklist); [2021 checklist page](https://neurips.cc/Conferences/2021/PaperInformation/PaperChecklist)
- NeurIPS Datasets & Benchmarks track, post dated 10 Mar 2025: "(1) Single-blind submissions, (2) Required dataset and benchmark code submission", and "requiring dataset authors to ensure that their datasets are properly hosted". — [NeurIPS blog](https://blog.neurips.cc/?p=1241)
- ICLR 2026: "Authors are strongly encouraged to include a paragraph-long Reproducibility Statement" (optional); "We encourage all authors to submit code"; "reviewers are encouraged, but not required to review supplementary material". — [ICLR 2026 Author Guide](https://iclr.cc/Conferences/2026/AuthorGuide)
- Papers with Code ML Code Completeness Checklist (five items: "Specification of dependencies", "Training code", "Evaluation code", "Pre-trained models", "README file including table of results accompanied by precise commands"), derived from NeurIPS 2019 repos; "Available now, the ML Code Completeness Checklist is part of the official NeurIPS 2020 code submission process" (Meta AI blog, 8 Apr 2020); use is "recommended (but not mandatory)". — [GitHub releasing-research-code](https://github.com/paperswithcode/releasing-research-code); [Meta AI blog](https://ai.meta.com/blog/new-code-completeness-checklist-and-reproducibility-updates)

**Summary table (as of 2026; "n/s" = not stated in sources reachable)**

| Venue | Code deposit | Data deposit | DAS mandatory | Reviewer code access | Effective date |
|---|---|---|---|---|---|
| Remote Sensing of Environment (Elsevier) | Encouraged (via Research Elements / SoftwareX) | Recommended (Option B/C class; exact option unverified) | Likely (statement appears in "Research data" section; per CASRAI "most Elsevier journals require a DAS") | Not provided for | n/s |
| ISPRS J. Photogramm. RS (Elsevier) | Encouraged | Journal option unverified (blocked) | Likely, unverified | Not provided for | n/s |
| Int. J. Appl. Earth Obs. Geoinf. (Elsevier) | Encouraged | Unverified | Likely, unverified | Not provided for | n/s |
| Computers and Electronics in Agriculture (Elsevier) | Encouraged | Unverified | Likely, unverified | Not provided for | n/s |
| Agricultural and Forest Meteorology (Elsevier) | Encouraged | Unverified | Likely, unverified | Not provided for | n/s |
| Science of Remote Sensing (Elsevier, OA) | Encouraged | "Deposit recommended" (CIRAD) | Likely, unverified | Not provided for | n/s |
| IEEE TGRS | Encouraged (IEEE-wide; Code Ocean optional) | Encouraged (IEEE DataPort/figshare/Zenodo) | No | Not provided for | Code Ocean link from 2017 |
| IEEE JSTARS | "Optional Code Ocean Submission" | Encouraged | No | Not provided for | 2017+ |
| IEEE GRSL | Encouraged; "Deposit recommended" | Encouraged | No | Not provided for | 2017+ |
| IGARSS | Not mentioned | Not mentioned | No | No | — |
| Remote Sensing (MDPI) | Not required | Recommended | Yes, every article | Not provided for | date not found |
| Nature Communications | Statement mandatory for central custom code; code must be shared with editors/reviewers for code peer review | Minimum dataset must be available; DAS mandatory | Yes | Yes | DAS trial Mar 2016, mandatory Sep 2016–early 2017 |
| Scientific Data | Code availability statement mandatory; code must be available to reviewers | Mandatory deposit in approved repository | Yes | Yes | Journal launched 2014 (date of code clause n/s) |
| Earth System Science Data | Not required | Mandatory with DOI at submission | Yes ("Data availability" section) | Yes (anonymous review access to data) | n/s |
| GIScience & Remote Sensing (T&F) | Not required | Tier unverified (registry says deposit required) | Depends on tier (Basic tier: encouraged) | Not provided for | OA from 2022 |
| CVPR / EarthVision | Optional, "expected but not required" for accepted papers | Dataset contributions public by camera-ready | No | Optional | 2024–2025 wording |
| NeurIPS main track | Checklist question; "No" acceptable | Same | Checklist mandatory (2021+) | Optional | 2021 |
| NeurIPS Datasets & Benchmarks | Required (2025) | Required hosting | Yes | Yes | 2025 |
| ICLR | Optional, encouraged | Optional | No (optional reproducibility statement) | Optional | — |

### Inferences
- For transfer papers published 2019–2025 in RSE, ISPRS JPRS, JAG, CEA, TGRS, JSTARS, GRSL or MDPI Remote Sensing, a missing or vague description of cloud/gap handling cannot be resolved from a mandated repository; only voluntarily linked GitHub/Code Ocean repos exist, and the fraction of papers with such links is a property of the paper, not the venue.
- The data availability statement is the one artefact that is mandatory at MDPI (all years in scope) and at Springer Nature (2016/17 onward), and present at most Elsevier journals; it usually names the imagery source (GEE collection, Copernicus Hub, Planet) and sometimes the preprocessing product (e.g., monthly composites), which is enough to re-derive cloud conditions by the procedures in KQ2 even when code is absent.
- Venues where a reviewer could have seen the code are Nature Communications, Scientific Data, ESSD (data only) and NeurIPS D&B 2025; everywhere else reviewers worked from the manuscript, so a reviewer-stage check of missingness handling was structurally unavailable.

### Gaps
- The exact Elsevier research-data option (A–E) adopted by RSE, ISPRS JPRS, JAG, CEA, AFM and Science of Remote Sensing could not be read: ScienceDirect guide-for-authors pages returned HTTP 403 / captcha for both WebFetch and a browser-UA curl. Needs a manual read of each "Research data" subsection.
- The date MDPI made the DAS mandatory was not found; the September 2025 MDPI instructions update does not mention it.
- GIScience & Remote Sensing's T&F tier is unconfirmed (journal page blocked; only a registry claim).
- IEEE DataPort is recommended on IEEE pages found by search but the Author Center page fetched names only figshare, Zenodo and Dryad; the launch date of the Code & Datasets tab was not found (partnership page is dated 14 Mar 2017).

---

## KQ2. What is the fastest API or GEE query to get per-scene cloudy percentage for a tile and date range, and how does one establish where the cloud/shadow gaps fell?

### Takeaway
Fastest with no account: a single GET to AWS Earth Search (`/v1/search?collections=sentinel-2-l1c&bbox=…&datetime=…&fields=…`) returns `eo:cloud_cover` and the MGRS tile per granule for any period since mid-2015 and was verified live for the Western Cape in 2017; the Copernicus Data Space OData endpoint does the same with a `cloudCover` attribute filter, and GEE gives the richest per-scene fields (`CLOUDY_PIXEL_PERCENTAGE`, `CLOUDY_SHADOW_PERCENTAGE`, `HIGH/MEDIUM_PROBA_CLOUDS_PERCENTAGE`, `THIN_CIRRUS_PERCENTAGE`, `NODATA_PIXEL_PERCENTAGE`) plus pixel-level SCL, s2cloudless and Cloud Score+ for mapping *where* the gaps fell inside a field footprint.

### Cited Findings

**Sentinel-2 scene metadata fields (GEE, `COPERNICUS/S2_SR_HARMONIZED`)**

- Properties: `CLOUDY_PIXEL_PERCENTAGE` "Granule-specific cloudy pixel percentage taken from the original metadata"; `CLOUD_COVERAGE_ASSESSMENT` "Cloudy pixel percentage for the whole archive that contains this granule"; `CLOUDY_SHADOW_PERCENTAGE` "Percentage of pixels classified as cloud shadow"; `DARK_FEATURES_PERCENTAGE`; `HIGH_PROBA_CLOUDS_PERCENTAGE`; `MEDIUM_PROBA_CLOUDS_PERCENTAGE`; `THIN_CIRRUS_PERCENTAGE`; `NODATA_PIXEL_PERCENTAGE`; `VEGETATION_PERCENTAGE`. L2A availability 2017-03-28 onward. — [GEE S2_SR_HARMONIZED](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED)
- SCL classes: 1 Saturated/defective, 2 Dark Area Pixels, 3 Cloud Shadows, 4 Vegetation, 5 Bare Soils, 6 Water, 7 Clouds Low Probability/Unclassified, 8 Clouds Medium Probability, 9 Clouds High Probability, 10 Cirrus, 11 Snow/Ice. `QA60` bit 10 opaque cloud, bit 11 cirrus; QA60 "contained rasterized cloud mask polygons until 2022-01-25" and is masked 2022-01-25 to 2024-02-28. — same page.
- s2cloudless: `COPERNICUS/S2_CLOUD_PROBABILITY`, band `probability` (0–100), "created with the sentinel2-cloud-detector library (using LightGBM)", 10 m, from 2015-06-27; join on `system:index`. — [GEE S2_CLOUD_PROBABILITY](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_CLOUD_PROBABILITY)
- Cloud Score+: `GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED`, bands `cs` and `cs_cdf` (0 = occluded, 1 = clear); "values between 0.50 and 0.65 generally work well"; link with `.linkCollection(csPlus, ['cs_cdf'])`; from 2015-06-27 (backfill ongoing); citation Pasquarella et al. 2023 CVPRW. — [GEE Cloud Score+](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_CLOUD_SCORE_PLUS_V1_S2_HARMONIZED)

**Copernicus Data Space (no account for catalogue search)**

- OData, verbatim from the docs: `https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=Collection/Name eq 'SENTINEL-2' and Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and att/OData.CSC.DoubleAttribute/Value le 40.00) and ContentDate/Start gt 2022-01-01T00:00:00.000Z and ContentDate/Start lt 2022-01-03T00:00:00.000Z&$top=10`; add `Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/OData.CSC.StringAttribute/Value eq 'S2MSI2A')` and `OData.CSC.Intersects(area=geography'SRID=4326;POLYGON((…))')` for footprint; docs advise constraining `ContentDate/Start` for speed. — [CDSE OData docs](https://documentation.dataspace.copernicus.eu/APIs/OData.html)
- STAC: base `https://stac.dataspace.copernicus.eu/v1/`, collections `sentinel-2-l2a`, `sentinel-2-l1c`; CQL2 example `collections/sentinel-2-l2a/items?filter-lang=cql2-text&filter=eo:cloud_cover<15`; queryables at `/v1/collections/sentinel-2-l2a/queryables`. — [CDSE STAC docs](https://documentation.dataspace.copernicus.eu/APIs/STAC.html)

**Microsoft Planetary Computer**

- `sentinel-2-l2a` items carry (verified on a live item): `eo:cloud_cover`, `s2:mgrs_tile`, `s2:cloud_shadow_percentage`, `s2:high_proba_clouds_percentage`, `s2:medium_proba_clouds_percentage`, `s2:thin_cirrus_percentage`, `s2:nodata_pixel_percentage`, `s2:vegetation_percentage`, `s2:not_vegetated_percentage`, `s2:water_percentage`, `s2:snow_ice_percentage`, `s2:unclassified_percentage`, `s2:degraded_msi_data_percentage`, `s2:processing_baseline`, `s2:product_type`, plus SCL as an asset; collection temporal extent starts 2015-06-27. — [PC sentinel-2-l2a items](https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-2-l2a/items?limit=1); [collection JSON](https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-2-l2a)
- `landsat-c2-l2` items carry `eo:cloud_cover`, `landsat:cloud_cover_land`, `landsat:wrs_path/row`, `landsat:collection_category` (T1/T2) and a `qa_pixel` asset ("Collection 2 Level-1 Pixel Quality Assessment Band"). — [PC landsat-c2-l2 items](https://planetarycomputer.microsoft.com/api/stac/v1/collections/landsat-c2-l2/items?limit=1)

**AWS Earth Search (Element 84), verified live**

- GET `https://earth-search.aws.element84.com/v1/search?collections=sentinel-2-l1c&bbox=19.9,-34.4,20.3,-34.0&datetime=2017-05-01T00:00:00Z/2017-09-30T23:59:59Z&limit=100&fields=id,properties.datetime,properties.eo:cloud_cover,properties.grid:code,-geometry,-assets,-links` returned 167 matched granules (`context.matched`), each with `datetime`, `grid:code` (e.g. `MGRS-34HDH`) and `eo:cloud_cover`. — [Earth Search query](https://earth-search.aws.element84.com/v1/search?collections=sentinel-2-l1c&bbox=19.9,-34.4,20.3,-34.0&datetime=2017-05-01T00:00:00Z/2017-09-30T23:59:59Z&limit=100&fields=id,properties.datetime,properties.eo:cloud_cover,properties.grid:code,-geometry,-assets,-links); collection extent from 2015-06-27 — [Earth Search sentinel-2-l1c](https://earth-search.aws.element84.com/v1/collections/sentinel-2-l1c)
- POST form with pagination used for the harvest in this folder (`earthsearch_harvest.py`): body `{"collections":["sentinel-2-l1c"],"bbox":[...],"datetime":"start/end","limit":500,"fields":{"include":["id","properties.datetime","properties.eo:cloud_cover","properties.grid:code"],"exclude":["geometry","assets","links","bbox"]}}`, follow `links[rel=next].body`. 33 points × 7 years ran in a few minutes with no key.

**Landsat Collection 2**

- `LANDSAT/LC08/C02/T1_L2` `QA_PIXEL` bits: 0 Fill, 1 Dilated Cloud, 2 Cirrus (high confidence), 3 Cloud, 4 Cloud Shadow, 5 Snow, 6 Clear ("1: Cloud and Dilated Cloud bits are not set"), 7 Water, 8–9 Cloud Confidence, 10–11 Cloud Shadow Confidence, 12–13 Snow/Ice Confidence, 14–15 Cirrus Confidence; scene properties `CLOUD_COVER` and `CLOUD_COVER_LAND` ("Percentage cloud cover over land (0-100), -1 = not calculated"). — [GEE LC08 C02 T1_L2](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2)

**MODIS cloud climatologies**

- EarthEnv cloud (Wilson & Jetz 2016): cloud flag from "bit 10 of the daily daytime surface reflectance product 'state 1 km'" of MOD09GA (Terra ~10:30) and MYD09GA (Aqua ~13:30), Feb 2000–Mar 2014, ~1 km; monthly cloud frequency = "proportion of days with a positive cloud flag"; validated against 1,558 stations (captures 78 % of variance in monthly mean cloud); data at earthenv.org/cloud. — [Wilson & Jetz 2016, PLOS Biol](https://pmc.ncbi.nlm.nih.gov/articles/PMC4816575/)
- GEE community assets: `projects/sat-io/open-datasets/gcc/MODCF_monthlymean` (ImageCollection), `MODCF_meanannual`, `MODCF_intraannualSD`, `MODCF_seasonality_*` (CC BY-NC 4.0). — [GEE community catalog, Global 1-km Cloud Cover](https://gee-community-catalog.org/projects/gcc/)
- `MODIS/061/MOD08_M3` monthly 1° atmosphere product, 2000-02 onward, is in GEE (cloud-fraction bands are in the band list, not shown on the summary page). — [GEE MOD08_M3](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD08_M3)

**Published clear-observation analyses**

- Whitcraft, Vermote, Becker-Reshef & Justice 2015 (RSE 156:438–447): 10–13 years of twice-daily MODIS Terra/Aqua cloud flags over the growing season; early/mid season "heavily and widely clouded"; for many important agricultural areas fewer than half of 8-day composites reach 70 % clear; morning acquisitions more often clear. — [NASA MODIS publication record](https://modis.gsfc.nasa.gov/sci_team/pubs/abstract_new.php?id=20047) (abstract text via search summary; publisher page blocked)
- Sudmanns, Tiede, Augustin & Lang (IJDE 2019/2020): metadata of all Sentinel-2 L1C scenes acquired in 2017; "higher acquisition frequency does not necessarily produce more cloud-free scenes"; identifies "systematically incorrect cloud-cover estimates over high, non-vegetated terrain"; results served via the EO-Compass web app. — [DOI 10.1080/17538947.2019.1572799](https://doi.org/10.1080/17538947.2019.1572799)
- Ju & Roy 2008 (RSE 112:1196–1211): 3 years of ETM+ cloud metadata; applications needing one cloud-free observation per year are largely unaffected, those needing a cloud-free observation in a given season or two per year are "severely affected". — [SDSU open repository record](https://openprairie.sdstate.edu/gsce_pubs/94)
- Lewińska et al. 2024, Data in Brief 57 (DOI 10.1016/j.dib.2024.111054): global daily usable-observation flags (cloud-, snow-, shade-free) for Landsat 1982–2025 and Sentinel-2 2015–2025 at 475,150 points on a 0.18° grid, with MCD12Q2 growing-season masks; Dryad DOI 10.5061/dryad.gb5mkkwxm. — [Dryad record](https://datadryad.org/dataset/doi:10.5061/dryad.gb5mkkwxm)
- Lewińska et al. 2024, Eur. J. Remote Sens. (DOI 10.1080/22797254.2024.2372855): compositing windows 5 days–1 year over Europe; sub-monthly composites "mostly limited to the combined Landsat and Sentinel-2 archives after 2015", and "in some geographies requires interpolation of up to 50% of data". — [Dryad record](https://datadryad.org/dataset/doi:10.5061/dryad.5tb2rbp94)
- Prudente et al. 2020, RSASE 20:100414 "Limitations of cloud cover for optical remote sensing of agricultural areas across South America": daily MODIS MCD19A2 C6 2000–2015, monthly cloud frequency grouped by crop-season quarters; strongest constraints for summer soybean/maize in parts of Brazil. — [DOI 10.1016/j.rsase.2020.100414](https://doi.org/10.1016/j.rsase.2020.100414)
- Eberhardt et al. 2016, Remote Sensing 8(3):219: MOD35 1 km, Jul 2000–Jun 2014, four Center-South Brazilian states; northern states (SP, PR) ~15 % clear-sky during Nov–Feb summer cropping, southern (RS) ~45 % with no seasonality; "the most problematic period … is between December and January (P5) when clear sky coverage tended to be below 50%". — [MDPI](https://www.mdpi.com/2072-4292/8/3/219)

**Recipe (GEE, per-scene table and per-field gap map for a label footprint)**

```javascript
// Per-scene metadata over the footprint, L2A (or use COPERNICUS/S2_HARMONIZED for L1C back to 2015)
var roi = ee.Geometry.Rectangle([20.78, -34.20, 21.04, -33.98]);  // AI4FoodSecurity holdout 34S_20E_259N
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(roi).filterDate('2017-05-01','2017-10-01');
print(s2.reduceColumns(ee.Reducer.toList(5), ['system:index','MGRS_TILE','CLOUDY_PIXEL_PERCENTAGE','CLOUDY_SHADOW_PERCENTAGE','NODATA_PIXEL_PERCENTAGE']));
// Pixel-level: fraction of scenes in which each pixel was SCL cloud/shadow/cirrus (3,8,9,10)
var occl = s2.map(function(im){ var scl = im.select('SCL'); return scl.eq(3).or(scl.gte(8).and(scl.lte(10))).rename('occl'); }).mean().clip(roi);
// Or Cloud Score+: per-pixel count of clear observations (cs_cdf >= 0.6)
var cs = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED');
var clearN = s2.linkCollection(cs,['cs_cdf']).map(function(im){ return im.select('cs_cdf').gte(0.6); }).sum().clip(roi);
// Reduce to the label polygons (fields) to get per-field clear-observation counts by month.
```
Sources for every identifier: the GEE catalog pages cited above.

### Inferences
- Granule-level `eo:cloud_cover` / `CLOUDY_PIXEL_PERCENTAGE` is the right *first* check (seconds, no account) but it is tile-wide (~110 km); whether the holdout *fields* were occluded on a given date requires the SCL / Cloud Score+ pixel reduction over the label footprint. Both train and holdout AI4FoodSecurity regions sit largely in MGRS 34HDH, so granule-level stats cannot separate them; only the pixel-level reduction can.
- The L1C collections (`COPERNICUS/S2_HARMONIZED`, Earth Search `sentinel-2-l1c`, CDSE `sentinel-2-l1c`) are the only ones that cover pre-2017-03 and the full 2017 season for Africa, since global L2A/SCL starts 2017-03-28 in GEE and ESA's global L2A later elsewhere; Cloud Score+ is being backfilled to 2015 and covers 2017.
- Because 2017 metadata cloud estimates came from the older ESA L1C algorithm (Sudmanns flags terrain-related errors; QA60 polygons were coarse), audits of 2017-era papers should prefer s2cloudless or Cloud Score+ recomputed today over the original QA60/metadata the paper used.

### Gaps
- Whitcraft et al. 2015 abstract text was not retrievable from a primary page (ScienceDirect blocked, NASA pub pages failed TLS/DNS); the "fewer than half of 8-day composites reach 70 % clear" statement comes from a search summary and should be verified against the paper.
- The EO-Compass URL was not found by search; it is given in the Sudmanns paper.
- King et al. 2013 (TGRS, MODIS cloud climatology) and Mercury et al. 2012 (RSE, HyspIRI cloud study) abstracts were not retrievable (closed access; API records elided).
- Earth Search's documentation of how `eo:cloud_cover` is derived for `sentinel-2-l1c` (it mirrors the ESA granule `CLOUDY_PIXEL_PERCENTAGE` in `tileInfo.json`) was not fetched; this is an inference from the ESA metadata definition.

---

## KQ3. Which common target regions have growing seasons that are climatologically cloudier than the source regions typically paired with them?

### Takeaway
Measured as the share of Sentinel-2 L1C granules with < 20 % cloud (2018–2024, this harvest), the cloudiest growing seasons among the usual targets are the Guinea-savanna/monsoon belts (Mato Grosso Oct–Mar 0.13, Nigeria Kaduna and Ghana Tamale May–Oct 0.14–0.15, Sichuan 0.15, Yunnan 0.17, Andhra Pradesh kharif 0.19, Ethiopian meher 0.26, Sahel Jun–Oct 0.31–0.32, Tanzania unimodal season 0.32), while the usual European *sources* (Brittany, Denmark, Bavaria, Brandenburg, Netherlands, Belgium: 0.23–0.27 Apr–Sep) are themselves cloudy but with winter, not growing-season, maxima; targets that are *clearer* than European sources include the Western Cape winter (0.49–0.52 May–Oct), Catalonia (0.53), Punjab (0.53 kharif, 0.55 rabi), Iran (0.65), the WA wheatbelt (0.64), Central Valley (0.81) and the Pampas (0.52).

### Cited Findings

**Method for the computed column (reproducible; script and CSV in this folder)**

- For each representative cropland point a 0.04° box was searched in AWS Earth Search `sentinel-2-l1c` for 2018-01-01 to 2024-12-31; per month the number of granules, median `eo:cloud_cover`, and fraction of granules with `eo:cloud_cover` < 20 % were computed. Granule counts of ~1200 (one tile, one orbit), ~600 (one tile, 2018–2024 at 5-day revisit but one orbit) or ~2400 (two overlapping tiles/orbits) reflect tiling, not data loss. Cloud fraction is granule-wide (100 × 100 km) metadata, so it is a conservative proxy: a field can be clear inside a 40 %-cloudy granule. — computed here from [Earth Search](https://earth-search.aws.element84.com/v1/collections/sentinel-2-l1c); metadata definition per [GEE S2 catalog](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED)

**Regional table** ("P(clear granule)" = mean monthly fraction of granules < 20 % cloudy over the stated season; "worst month" = lowest such fraction; cloudiest months by median granule cloud %)

| Region (point) | Season used | P(clear granule) in season | Worst month | Cloudiest months (median cloud %) | Published corroboration |
|---|---|---|---|---|---|
| Brittany, FR (Rennes) | Apr–Sep | 0.27 | May 0.22 | Dec, Feb, Jan (78–82 %) | Atlantic Europe: sub-monthly composites need combined Landsat+S2 and up to 50 % interpolation — [Lewińska 2024](https://datadryad.org/dataset/doi:10.5061/dryad.5tb2rbp94) |
| Northern France (Somme) | Apr–Sep | 0.23 | Aug 0.19 | Feb, Dec, Jan (79–86 %) | same |
| Denmark (Viborg) | Apr–Sep | 0.26 | Aug 0.22 | Feb, Dec, Nov (86–89 %) | same |
| Austria (Marchfeld) | Apr–Sep | 0.36 | May 0.16 | Dec, Jan, Nov (83–89 %) | same |
| Bavaria (Straubing) | Apr–Sep | 0.27 | May 0.12 | Jan, Nov, Dec (88–96 %) | same |
| Brandenburg (Prenzlau) | Apr–Sep | 0.27 | May 0.22 | Jan, Nov, Dec (93–94 %) | same |
| Netherlands (Flevoland) | Apr–Sep | 0.24 | Aug 0.20 | Feb, Jan, Oct (81–93 %) | same |
| Belgium (Hesbaye) | Apr–Sep | 0.23 | May 0.19 | Feb, Jan, Oct (85–91 %) | same |
| Catalonia (Lleida) | Mar–Aug | 0.53 | Apr 0.23 | Apr, Nov, Dec (44–56 %) | Mediterranean region least constrained in Europe — [Lewińska 2024](https://datadryad.org/dataset/doi:10.5061/dryad.5tb2rbp94) |
| Western Cape, Overberg (Caledon) | May–Oct | 0.49 | Jun 0.40 | Mar, Jul, Jun (38–41 %) | "winter rainfall Mediterranean climate of southwestern South Africa" has strong intra-annual cloud seasonality — [Wilson & Jetz 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC4816575/); Stellenbosch thesis notes "frequent overcast conditions" — [SUNScholar](https://scholar.sun.ac.za/handle/10019.1/106943) |
| Western Cape, Swartland (Malmesbury) | May–Oct | 0.52 | Jul 0.45 | Jul, May, Jun (26–28 %) | same |
| Western Cape, actual AI4FoodSecurity holdout centre (20.91 E, 34.09 S) | May–Oct | 0.49 (Jun 0.55, Jul 0.56, Aug 0.59, Sep 0.31) | Sep 0.31 | Sep, Apr, Mar (33–42 %) | computed here (see KQ4) |
| Kenya western (Bungoma) | Mar–Aug long rains | 0.62 | Apr 0.44 | Nov, Apr, Feb (19–34 %) | S2 cloud-probability layer "does not reliably identify all clouds" in western Kenya 2019 — [Kerner et al. 2020](https://arxiv.org/pdf/2004.03023) |
| Kenya Rift (Nakuru) | Mar–Aug | 0.44 | Jun 0.30 | Nov, Jun, Aug (28–37 %) | same |
| Kenya short rains (both points) | Oct–Dec | 0.41–0.56 (Bungoma), 0.32–0.51 (Nakuru) | Nov | — | — |
| Tanzania southern highlands (Mbeya) | Dec–May unimodal | 0.32 | Jan 0.08 | Jan, Mar, Dec (46–70 %) | — |
| Nigeria (Kaduna) | May–Oct | 0.14 | Jun 0.02 | May, Jul, Jun (83–93 %) | SE Nigeria: mean 6.3 cloud-free S2 observations per pixel per 7-month season, mean cloud fraction ~84 %, peak 90.8 % in June (preprint) — [Research Square](https://www.researchsquare.com/article/rs-10032387/v1) |
| Ghana (Tamale) | May–Oct | 0.15 | Jul 0.02 | Aug, Jul, Jun (76–90 %) | — |
| Ethiopia (Arsi) | Jun–Nov meher | 0.26 | Jul 0.05 | Jul, Aug, Sep (39–59 %) | more pixels interpolated during the Highlands rainy season (Jun 1–Oct 1) — [arXiv 2202.04239](https://arxiv.org/pdf/2202.04239) |
| Mali (Ségou) | Jun–Oct | 0.32 | Aug 0.20 | Aug, Jun, Jul (63–71 %) | Sahel shows high intra-annual cloud variability — [Wilson & Jetz 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC4816575/) |
| Senegal (Kaolack) | Jun–Oct | 0.31 | Aug 0.18 | Aug, Sep, Jul (44–56 %) | same |
| Indian Punjab (Ludhiana) | Jun–Oct kharif; Nov–Apr rabi | 0.53; 0.55 | Aug 0.34; Jan 0.33 | Jan, Jul, Aug (38–67 %) | "Cloud intra-annual variability is highest over the monsoonal biomes of India" — [Wilson & Jetz 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC4816575/) |
| Andhra Pradesh (Guntur) | Jun–Nov kharif | 0.19 | Jul 0.06 | Jul, Aug, Sep (86–92 %) | same |
| Northeast China (Songnen, Harbin) | May–Sep | 0.38 | Jul 0.30 | Feb, Aug, Jul (53–62 %) | NE China has "sufficient valid observations" for annual Landsat analyses — [Zhou, Dong & Liu](https://researchers.westernsydney.edu.au/en/publications/are-there-sufficient-landsat-observations-for-retrospective-and-c/); in some years "no good-quality Landsat data can be obtained during an entire cropping cycle" — [Liu et al. 2020 RSE](https://www.ceom.ou.edu/media/docs/upload/Liu_Luo_2020_RSE_CroppingIntensit_China_sz.pdf) |
| North China Plain (Hengshui) | Mar–Sep | 0.41 (Jun–Sep maize 0.25–0.43) | Jul 0.25 | Jul, Feb, Jan (49–71 %) | same |
| Yunnan (Qujing) | May–Oct | 0.17 | Jun 0.12 | Jun, Sep, Jul (71–78 %) | valid observations "relatively lower in southern China (6.4 observations), especially in Yunnan-Guizhou Plateau" vs 17.4 in cold temperate zone — [Liu et al. 2022, RS 14:3134](https://www.mdpi.com/2072-4292/14/13/3134) |
| Sichuan (Chengdu plain) | May–Oct | 0.15 | Oct 0.07 | Oct, Jan, Dec (97–99 %) | Sichuan Basin named as "cloud-prone region… without sufficient optical observation data" — [Liu et al. 2022](https://www.mdpi.com/2072-4292/14/13/3134) |
| Iran (Fars, Marvdasht) | Nov–Jun wheat | 0.65 | Jan 0.43 | Jan, Mar, Feb (10–32 %) | — |
| US Corn Belt (Ames, IA) | May–Oct | 0.42 | May 0.30 | Jan, Mar, Dec (79–88 %) | US early/mid-season clouded; <50 % of 8-day composites 70 % clear in many areas — [Whitcraft et al. 2015](https://modis.gsfc.nasa.gov/sci_team/pubs/abstract_new.php?id=20047) |
| US Central Valley (Fresno) | Apr–Oct | 0.81 | Apr 0.60 | Dec, Jan, Mar (21–63 %) | — |
| Argentina Pampas (Pergamino) | Oct–Apr summer; Jun–Dec wheat | 0.52; 0.43–0.58 | Jan 0.42 | Jan, Jun, Oct (24–53 %) | cloud limits optical wheat/barley mapping in the austral Pampas (SAR outperformed) — search summary of a 2021 Coronel Rosales study, [SciELO](https://www.scielo.org.mx/pdf/igeo/n104/2448-7279-igeo-104-e60173.pdf) |
| Brazil Mato Grosso (Sorriso) | Oct–Mar soy; Feb–Jun safrinha | 0.13; 0.10–0.73 | Nov 0.02 | Nov, Feb, Jan (80–92 %) | northern Center-South states ~15 % clear sky Nov–Feb; Dec–Jan "below 50%" — [Eberhardt et al. 2016](https://www.mdpi.com/2072-4292/8/3/219); summer crops most constrained — [Prudente et al. 2020](https://doi.org/10.1016/j.rsase.2020.100414) |
| Brazil Rio Grande do Sul (Cruz Alta) | Oct–Apr; Jun–Nov wheat | 0.48; 0.25–0.48 | Oct 0.37; Jun 0.25 | Jun, Nov, Oct (44–74 %) | RS ~45 % clear sky with "no significant clear sky seasonality" — [Eberhardt et al. 2016](https://www.mdpi.com/2072-4292/8/3/219) |
| Australia Victoria (Horsham, Wimmera) | May–Nov | 0.39 | Sep 0.26 | Nov, Sep, Oct (42–56 %) | — |
| Australia WA wheatbelt (Merredin) | May–Nov | 0.64 | Nov 0.58 | Sep, Oct, Apr (6–9 %) | — |

- Europe-wide context: Landsat+Sentinel-2 composites "ensured overall 50-62% data coverage for each calendar year, and ~75% afterwards [2015+], with further increase to ~82% when Landsat and Sentinel-2 were combined"; monthly composites were "challenging without joining Landsat and Sentinel-2 archives after 2015, and beyond the Mediterranean biogeographical region". — search summary of [Lewińska et al. 2024 preprint](https://www.preprints.org/manuscript/202308.2174/v2)
- A Guangdong (subtropical China) rice study found usable cloud-free observation probability of 52.1 % for Landsat 8 and 47.7 % for Sentinel-2 (search summary; URL attribution to [MDPI RS 13:2961](https://www.mdpi.com/2072-4292/13/15/2961) unverified).

### Inferences
- Pairings in which the target growing season is climatologically cloudier than the source (and so a missingness shift is the default, not the exception): any European or US source → Nigeria/Ghana, Ethiopia (meher), Sahel, Andhra Pradesh kharif, Yunnan/Sichuan, Mato Grosso summer, Tanzania unimodal season; European source (Apr–Sep ≈0.25) → Argentina summer or RS summer is roughly neutral to slightly *clearer*; European → Western Cape winter is markedly *clearer* at the granule level (0.49–0.52 vs 0.23–0.27) but has its own two cloud pulses (Jun and Aug–Sep) during the critical tillering/heading window, which is exactly where our 2017 gaps fall (KQ4).
- The shape of missingness also differs: European sources have gaps spread through the season with winter maxima; monsoon/Sahel targets have a single near-total blackout (June–September granule medians 60–93 %) so that most "growing-season" features in the target come from a few shoulder-month scenes, and interpolation/compositing choices dominate the feature distribution.
- Granule-level P(clear) overstates occlusion for small fields in partly cloudy scenes and understates it where the ESA mask misses thin cloud/haze (Sudmanns' terrain error; Kerner et al.'s under-detection), so the table ranks regions reliably but the absolute values should be re-derived at pixel level (Cloud Score+ or SCL) for any specific audit.

### Gaps
- No published per-region table of Sentinel-2 clear-observation probability exists for most of the listed targets (Denmark, Austria, Catalonia, Belgium, Netherlands, Kenya/Tanzania, Ghana, Mali/Senegal, Punjab, NE China, Iran, Central Valley, Victoria, WA); the values above for those regions rest only on the computation in this folder.
- Sudmanns et al. 2019 contains global 2017 maps of scene availability and cloudiness by region but the full text (Taylor & Francis) was blocked; its regional numbers could not be quoted.
- EarthEnv monthly cloud frequency values were not extracted for the regions (would need GEE or large GeoTIFF downloads); the `MODCF_monthlymean` asset listed in KQ2 is the way to add that column.
- Kenya/Tanzania and Indian Punjab observation counts per season were not found in the literature (Copernicus4GEOGLAM reports exist but do not report valid-observation counts in the retrieved snippets).

---

## KQ4. What can be established about cloud conditions over the AI4FoodSecurity Western Cape tiles in winter 2017?

### Takeaway
The dataset's label extents are at 20.52–21.04 °E, 33.98–34.41 °S (MGRS 34HDH for both train tiles, 34HDH + 34HDG for 34S_19E_258N, 34HDH + 34HEH for the holdout), and Earth Search metadata for 2017 shows two winter cloud pulses, June (median granule cloud 51–59 %, 3 of 10 holdout granules < 20 %) and 11–21 August (three consecutive overpasses 58–96 % cloudy), with a further 15-day gap 10–25 September; 2017 winter was cloudier than the 2018–2024 norm at the same points (June P(clear) 0.33–0.40 vs 0.50–0.55; August 0.33–0.50 vs 0.58–0.59), while train and holdout granule cloudiness were nearly identical on every shared date, so any train/holdout missingness difference must be sub-tile and needs the pixel-level recipe in KQ2.

### Cited Findings

- Dataset extents read from the label collections on disk: `34S_19E_259N` bbox [20.5224, −34.1968, 20.7835, −33.9796]; `34S_19E_258N` [20.5212, −34.4133, 20.7829, −34.1961]; `34S_20E_259N` [20.7829, −34.1970, 21.0434, −33.9804]; temporal extent 2017-01-01 to 2017-12-31. — local files `ref_fusion_competition_south_africa_{train,test}_labels/collection.json` (Radiant Earth / Western Cape Dept. of Agriculture, DOI 10.34911/rdnt.j0co8q, [source.coop](https://source.coop/radiantearth/south-africa-crops-competition))
- Granules intersecting them in 2017 (Earth Search `sentinel-2-l1c`): 34S_19E_259N → MGRS-34HDH only (110 granules); 34S_19E_258N → 34HDG + 34HDH (214); 34S_20E_259N → 34HDH + 34HEH (164). — computed here, `earthsearch_s2_l1c_cloudcover_SA_tiles_2017.csv`
- Holdout region 2017 by month (n granules, median cloud %, fraction < 20 %): May 10, 19.4, 0.50; Jun 10, 59.0, 0.20; Jul 19, 2.0, 0.74; Aug 18, 54.7, 0.39; Sep 14, 8.2, 0.64; Oct 19, 4.9, 0.74. Train 34S_19E_258N: May 14, 23.1, 0.43; Jun 13, 51.4, 0.23; Jul 24, 14.2, 0.58; Aug 24, 61.9, 0.33; Sep 18, 31.1, 0.50; Oct 25, 3.2, 0.72. — same CSV
- Dates May–Sep 2017 on which both the train tile 34HDH and the holdout-only tile 34HEH were acquired, with (34HDH %, 34HEH %): 05-08 (4.7, 1.9), 05-18 (21.0, 1.6), 05-28 (84.5, 83.6), 06-07 (51.4, 47.2), 06-17 (93.5, 94.6), 06-27 (68.3, 94.1), 07-02 (1.5, 0.0), 07-07 (2.0, 0), 07-12 (0, 0), 07-17 (0, 0), 07-22 (4.4, 4.1), 07-27 (0, 0), 08-06 (0.9, 1.4), 08-11 (89.2, 81.5), 08-16 (77.0, 78.4), 08-21 (58.4, 61.6), 08-26 (0.4, 0), 08-31 (7.1, 10.9), 09-05 (3.9, 0.1), 09-10 (4.1, 8.3), 09-15 (98.4, 86.3), 09-25 (95.0, 71.9), 09-30 (1.9, 3.8). No date had one tile < 20 % and the other ≥ 50 %. — computed here from `sa2017_raw.jsonl` (scratch; summarised in the CSV)
- Granules with < 20 % cloud at the holdout centre (20.90–20.92 °E, 34.08–34.10 °S), May–Sep 2017: 05-08, 06-20, 06-30, 07-02, 07-07, 07-10, 07-12, 07-17, 07-22, 07-27, 08-06, 08-26, 08-31, 09-05, 09-10, 09-23, 09-30 (17 of 27 acquisitions). Longest runs without a < 20 % granule: 9 May–19 June (41 days), 7–25 August, 11–22 September. — computed here
- Climatology at the same holdout centre, 2018–2024 (n = 668): fraction < 20 % by month Jan 0.51, Feb 0.45, Mar 0.36, Apr 0.45, May 0.46, Jun 0.55, Jul 0.56, Aug 0.59, Sep 0.31, Oct 0.49, Nov 0.50, Dec 0.37. Train centre (20.64–20.66 °E, 34.29–34.31 °S), 2018–2024 (n = 1244): May 0.56, Jun 0.50, Jul 0.52, Aug 0.58, Sep 0.44. — computed here
- Independent corroboration of the June gap: the derived IEEE DataPort dataset for this project states that imagery was masked with s2cloudless plus a shadow mask and composited monthly, and that "June 2017 had no usable imagery". — [IEEE DataPort, South Africa Crop Comp](https://ieee-dataport.org/documents/south-africa-crop-comp)
- Acquisition density doubled from July 2017 (holdout: 10 granules in May and June, 19 in July), consistent with the Sentinel-2B ramp-up; 2017 therefore mixes a 10-day and a 5-day revisit regime within one season. — computed here (interpretation is an inference)
- Labels were collected by aerial and vehicle survey between May 2017 and March 2018; the challenge used S2 time series from the 2017 season. — [torchgeo dataset docs](https://torchgeo.readthedocs.io/en/latest/_modules/torchgeo/datasets/south_africa_crop_type.html); [Zindi competition page](https://zindi.africa/competitions/radiant-earth-spot-the-crop-challenge)

### Inferences
- At granule scale, the 2017 train and holdout regions saw the *same* cloud history; the "missingness shift" between them, if real, lives at the sub-tile scale (the holdout's eastern third in 34HEH and local cloud/shadow footprints inside 34HDH), which is exactly what the September-gap overlay in this repo's recent commits targets and what a per-field Cloud Score+ clear-count would quantify.
- The two winter blackouts fall on agronomically decisive windows for Western Cape winter cereals and canola (June: early vegetative; mid-August to late September: stem elongation to flowering), so monthly compositing with a missing June and a thin August/September is the dominant preprocessing artefact of this dataset, and 2017 was cloudier in those months than a typical year at the same points.
- Papers that trained on these tiles and reported only "cloud-masked monthly composites" in their data statement can be audited by re-running the KQ2 pixel recipe on the published label footprints, since the imagery provenance (Sentinel-2 L1C/L2A over 2017) is fixed by the dataset.

### Gaps
- The pixel-level (SCL / Cloud Score+) per-field clear-observation counts for the three label extents were not computed here: GEE credentials exist on this machine but the session's permission policy blocked inspecting them, and the repo's `light_learn/0_SA_ee_download_data_prep.py` has `ee.Initialize()` commented out with no project id, so the GEE recipe in KQ2 is left for the user to run.
- The Radiant Earth tile names (34S_19E_258N etc.) do not correspond to MGRS tiles or to the 19–20 °E longitudes they suggest; no documentation of the naming scheme was found.
- No published analysis of 2017 Western Cape cloud conditions beyond the IEEE DataPort note was found; the figures above are first-hand metadata harvests, not literature values.
