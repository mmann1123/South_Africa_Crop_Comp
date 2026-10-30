# You et al. 2021 Google Earth Engine scripts (preprocessing recipe followed by DACCN)

Source: You, N., Dong, J., Huang, J., Du, G., Zhang, G., He, Y., Yang, T., Di, Y., Xiao, X. (2021). The 10-m crop type maps in Northeast China during 2017-2019. Scientific Data 8:41, https://doi.org/10.1038/s41597-021-00827-9. Code availability statement: "JavaScript code used to generate the cropland layer and crop type maps are available from the figshare repository."

Files downloaded 28 October 2026 from figshare record 10.6084/m9.figshare.13090442.v1 (https://figshare.com/articles/figure/The_10-m_crop_type_maps_in_Northeast_China_during_2017-2019/13090442), licence CC BY 4.0, files "GEE code for crop classification" (https://ndownloader.figshare.com/files/25109540) and "GEE code for cropland classification" (https://ndownloader.figshare.com/files/25109537). Line endings converted from CRLF; content otherwise unchanged.

Why they are here: Wang, Feng, Zhang and Tian 2023 (DACCN, ISPRS JPRS 199, https://doi.org/10.1016/j.isprsjprs.2023.04.002) state in Section 2.4 "Following You et al. (2021), we then preprocessed the satellite imagery" and describe the same chain (cloud mask citing Oreopoulos et al. 2011, 10-day median composites, linear interpolation of gaps, Savitzky-Golay with a 7-composite window, Google Earth Engine). DACCN released no code, so this is the closest auditable implementation of its pipeline.

What the crop-classification script shows that neither paper states:
- The cloud mask is the Earth Engine "simple cloud score" (Hancher, Hewig, Housman) with threshold 0.2 on L1C TOA; the function is commented "simply the cloud_and_shadow_mask" but contains no shadow test.
- 10-day composites are medians of the masked collection; a composite with no valid observation is tagged `length` 0 and fully masked.
- Linear interpolation looks at most 9 composites (90 days) backwards and forwards for the nearest valid composite on each side (`LIC.slice(i-9, i)` and `LIC.slice(i+1, i+10)`), and the loop runs over composites 9 to 30 (DOY 90 to 300, i.e. April to October, 22 steps; DACCN reports 21).
- After interpolation every image is passed through `.unmask(0)`, so a pixel with no valid observation on one or both sides within 90 days, including the first and last composites of the season, is set to 0 in every band and index. The published description ("data gaps filled by linear interpolation") does not mention this zero fill.
- Savitzky-Golay smoothing is `s2Tools.sgsmooth(interpolated, bands, order=3, window_size=7)` from the private Earth Engine module `users/nanshany1993/common:sentinel2`, which is not in the figshare deposit; DACCN reports a quadratic polynomial.
- No per-pixel or per-region count of valid observations, empty composites or interpolated steps is computed or exported anywhere in either script.

The cropland script uses the same `cloudMask` with 60-day median composites between 20 April and 21 October and no interpolation.
