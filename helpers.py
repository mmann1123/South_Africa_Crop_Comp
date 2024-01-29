import ee

CLD_PRB_THRESH = 80
NIR_DRK_THRESH = 1
CLD_PRJ_DIST = 2
BUFFER = 90
SCALE = 10


def add_cloud_bands(img, CLD_PRB_THRESH):
    # Get s2cloudless image, subset the probability band.
    cld_prb = ee.Image(img.get("s2cloudless")).select("probability")

    # Condition s2cloudless by the probability threshold value.
    is_cloud = cld_prb.gt(CLD_PRB_THRESH).rename("clouds")

    # Add the cloud probability layer and cloud mask as image bands.
    return img.addBands(ee.Image([cld_prb, is_cloud]))


def add_shadow_bands(img, NIR_DRK_THRESH, CLD_PRJ_DIST):
    # Identify water pixels from the SCL band.
    not_water = img.select("SCL").neq(6)

    # Identify dark NIR pixels that are not water (potential cloud shadow pixels).
    SR_BAND_SCALE = 1e4
    dark_pixels = (
        img.select("B8")
        .lt(NIR_DRK_THRESH * SR_BAND_SCALE)
        .multiply(not_water)
        .rename("dark_pixels")
    )

    # Determine the direction to project cloud shadow from clouds (assumes UTM projection).
    shadow_azimuth = ee.Number(90).subtract(
        ee.Number(img.get("MEAN_SOLAR_AZIMUTH_ANGLE"))
    )

    # Project shadows from clouds for the distance specified by the CLD_PRJ_DIST input.
    cld_proj = (
        img.select("clouds")
        .directionalDistanceTransform(shadow_azimuth, CLD_PRJ_DIST * 10)
        .reproject(**{"crs": img.select(0).projection(), "scale": 100})
        .select("distance")
        .mask()
        .rename("cloud_transform")
    )

    # Identify the intersection of dark pixels with cloud shadow projection.
    shadows = cld_proj.multiply(dark_pixels).rename("shadows")

    # Add dark pixels, cloud projection, and identified shadows as image bands.
    return img.addBands(ee.Image([dark_pixels, cld_proj, shadows]))


def add_cld_shdw_mask(
    img, CLD_PRB_THRESH=CLD_PRB_THRESH, NIR_DRK_THRESH=NIR_DRK_THRESH, SCALE=SCALE
):
    # Add cloud component bands.
    img_cloud = add_cloud_bands(img, CLD_PRB_THRESH)

    # Add cloud shadow component bands.
    img_cloud_shadow = add_shadow_bands(img_cloud, NIR_DRK_THRESH, CLD_PRJ_DIST)

    # Combine cloud and shadow mask, set cloud and shadow as value 1, else 0.
    is_cld_shdw = (
        img_cloud_shadow.select("clouds").add(img_cloud_shadow.select("shadows")).gt(0)
    )

    # Remove small cloud-shadow patches and dilate remaining pixels by BUFFER input.
    # 20 m scale is for speed, and assumes clouds don't require 10 m precision.
    is_cld_shdw = (
        is_cld_shdw.focalMin(2)
        .focalMax(BUFFER * 2 / SCALE)
        .reproject(**{"crs": img.select([0]).projection(), "scale": SCALE})
        .rename("cloudmask")
    )

    # Add the final cloud-shadow mask to the image.
    return img_cloud_shadow.addBands(is_cld_shdw)


def apply_cld_shdw_mask(img):
    # Subset the cloudmask band and invert it so clouds/shadow are 0, else 1.
    not_cld_shdw = img.select("cloudmask").Not()

    # Subset reflectance bands and update their masks, return the result.
    return img.select("B.*").updateMask(not_cld_shdw)


def get_s2_sr_cld_col(aoi, start_date, end_date, product="S2_SR", CLOUD_FILTER=30):
    # Import and filter S2 SR.
    s2_sr_col = (
        ee.ImageCollection("COPERNICUS/" + product)
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", CLOUD_FILTER))
    )

    # Import and filter s2cloudless.
    s2_cloudless_col = (
        ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
    )

    # Join the filtered s2cloudless collection to the SR collection by the 'system:index' property.
    return ee.ImageCollection(
        ee.Join.saveFirst("s2cloudless").apply(
            **{
                "primary": s2_sr_col,
                "secondary": s2_cloudless_col,
                "condition": ee.Filter.equals(
                    **{"leftField": "system:index", "rightField": "system:index"}
                ),
            }
        )
    )


def get_s2A_SR_sr_cld_col(
    aoi, start_date, end_date, product="S2_SR_HARMONIZED", CLOUD_lte=30
):
    # Import and filter S2 SR.
    s2_sr_col = (
        ee.ImageCollection("COPERNICUS/" + product)
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", CLOUD_lte))
    )

    # Import and filter s2cloudless.
    s2_cloudless_col = (
        ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
    )

    # Join the filtered s2cloudless collection to the SR collection by the 'system:index' property.
    return ee.ImageCollection(
        ee.Join.saveFirst("s2cloudless").apply(
            **{
                "primary": s2_sr_col,
                "secondary": s2_cloudless_col,
                "condition": ee.Filter.equals(
                    **{"leftField": "system:index", "rightField": "system:index"}
                ),
            }
        )
    )


def addEVI(image):
    EVI = image.expression(
        "2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))",
        {
            "NIR": image.select("B8").divide(10000),
            "RED": image.select("B4").divide(10000),
            "BLUE": image.select("B2").divide(10000),
        },
    ).rename("EVI")

    return image.addBands(EVI)
