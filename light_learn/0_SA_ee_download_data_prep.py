# Download using google earth engine since required 1c atmospheric correction.

# USing https://code.earthengine.google.com/?scriptPath=users%2Fmmann1123%2Fdefault%3ASouthAfrica%2FSentinel1c_Atmos_Downloader


# # use geepy environment, run earthengine authenticate in commandline first
# # %%
# # requires https://cloud.google.com/sdk/docs/install
# # and https://developers.google.com/earth-engine/guides/python_install-conda


# import pendulum
# import ee
# import os

# import pandas as pd
# from helpers import *
# from ipygee import *
# import ipygee as ui
# import geopandas as gpd
# import geemap

# # ee.Authenticate()  # rerun if token expires

# ee.Initialize()
# import geetools
# from geetools import ui, cloud_mask, batch

# # # export clipped result in Tiff
# crs = "EPSG:32734"

# # desktop
# os.chdir(
#     "/home/mmann1123/extra_space/Dropbox/South_Africa_data/Projects/Agriculture_Comp/"
# )

# # %%
# # ## Get image bounds

# # ## Get image bounds
# bounds = gpd.read_file(r"bounds/totalbounds.gpkg")
# bbox = bounds.to_crs("epsg:4326").total_bounds
# site = ee.Geometry.Rectangle(bbox.tolist())


# # collection = (
# #     ee.ImageCollection("COPERNICUS/S2")
# #     .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", 50))
# #     .filterDate("2017-01-01", "2017-1-10")
# #     .select(["B4", "B3", "B2"])
# #     .filterBounds(site)
# # )
# collection = (
#     get_s2_sr_cld_col(
#         aoi=site,
#         start_date="2017-01-01",
#         end_date="2017-1-10",
#         product="S2",
#         CLOUD_FILTER=30,
#     ).select(["B4", "B3", "B2"])
#     # .median()
# )
# eprint(collection.getInfo())
# # %%
# # download using geemap
# geemap.ee_export_image(
#     collection,
#     # out_dir="~/Downloads",
#     filename="S2.tif",
#     scale=100,
#     crs=crs,
#     region=site,
#     file_per_band=True,
#     timeout=300,
#     proxies=None,
# )

# %%

# # Path to your local shapefile
# shapefile_path = r"bounds/totalbounds.gpkg"

# # Read the shapefile using geopandas
# gdf = gpd.read_file(shapefile_path)

# # Get the bounds of the shapefile
# roi_bounds = gdf.total_bounds  # Returns [minx, miny, maxx, maxy]

# # Create a rectangle geometry from the bounds
# site = ee.Geometry.Rectangle(roi_bounds.tolist())


# # Set parameters
# bands = ["B2", "B3", "B4", "B8"]
# scale = 10
# # date_pattern = "mm_dd_yyyy"  # dd: day, MMM: month (JAN), y: year
# folder = "SouthAfrica_Fields"

# # extra = dict(sat="Sen_TOA")
# CLOUD_lte = 75

# # %%
# # # Initialize the Earth Engine API
# # ee.Initialize()

# # # Path to your local shapefile
# # shapefile_path = r"bounds/totalbounds.gpkg"

# # # Read the shapefile using geopandas
# # gdf = gpd.read_file(shapefile_path)

# # # Get the bounds of the shapefile
# # roi_bounds = gdf.total_bounds  # Returns [minx, miny, maxx, maxy]

# # # Create a rectangle geometry from the bounds
# # roi_geometry = ee.Geometry.Rectangle(roi_bounds.tolist())

# # # Load an image collection
# # image_collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")

# # # Filter the image collection based on the ROI
# # filtered_collection = image_collection.filterBounds(roi_geometry).filterDate(
# #     "2017-04-01", "2017-12-30"
# # )

# # # Print the size of the filtered collection
# # print("Number:", filtered_collection.size().getInfo())


# # %% MONTHLY COMPOSITES EVI

# for year in list(range(2017, 2018)):
#     for month in list(range(8, 13, 1)):
#         print(year, month)
#         dt = pendulum.datetime(year, month, 1)

#         collection = get_s1C_SR_sr_cld_col(
#             aoi=site,
#             start_date=dt.start_of("month").strftime(r"%Y-%m-%d"),
#             end_date=dt.end_of("month").strftime(r"%Y-%m-%d"),
#             CLOUD_lte=CLOUD_lte,
#         )
#         print("number of images", collection.size().getInfo())
#         # %%
#         s2_sr = (
#             collection.map(add_cld_shdw_mask)
#             .map(apply_cld_shdw_mask)
#             .select(bands)
#             .map(addEVI)
#             .select(["EVI"])
#             .median()
#             .multiply(10000)
#             .clip(site)
#             .unmask(9999)
#         )
#         s2_sr = geetools.batch.utils.convertDataType("int16")(s2_sr)
#         # eprint(s2_sr)

#         img_name = f"SA_S2_SR_EVI_M_" + str(year) + "_" + str(month).zfill(2)
#         export_config = {
#             "scale": scale,
#             "maxPixels": 9000000000,
#             "driveFolder": folder,
#             "region": site,
#             "crs": crs,
#         }
#         task = ee.batch.Export.image(s2_sr, img_name, export_config)
#         task.start()


# ############################################################################################################
# # %% SWIR and Red Edge

# # Set parameters ("B5" not important in mini-study)
# bands = ["B2", "B6", "B11", "B12"]
# scales = [10, 20, 20, 20, 20]
# # date_pattern = "mm_dd_yyyy"  # dd: day, MMM: month (JAN), y: year
# folder = "SouthAfrica_Fields"


# # %% MONTHLY COMPOSITES
# for year in list(range(2017, 2018)):
#     for month in list(range(8, 13, 1)):
#         print(year, month)
#         dt = pendulum.datetime(year, month, 1)

#         collection = get_s2A_SR_sr_cld_col(
#             aoi=site,
#             start_date=dt.start_of("month").strftime(r"%Y-%m-%d"),
#             end_date=dt.end_of("month").strftime(r"%Y-%m-%d"),
#             CLOUD_lte=CLOUD_lte,
#         )

#         print("number of images", collection.size().getInfo())
#         # %%
#         for band, scale in zip(bands[0:1], scales[0:1]):
#             s2_sr = (
#                 collection.map(add_cld_shdw_mask)
#                 .map(apply_cld_shdw_mask)
#                 .select(band)
#                 .median()
#                 .clip(site)
#                 .unmask(9999)
#             )
#             s2_sr = geetools.batch.utils.convertDataType("int16")(s2_sr)
#             eprint(s2_sr)
#             # %%
#             # # export clipped result in Tiff
#             img_name = f"SA_S2_SR_{band}_M_{year}_{str(month).zfill(2)}"
#             export_config = {
#                 "scale": scale,
#                 "maxPixels": 9000000000,
#                 "driveFolder": folder,
#                 "region": site,
#                 "crs": crs,
#             }
#             task = ee.batch.Export.image(s2_sr, img_name, export_config)
#             task.start()


# ##################################################################
# # %% HSV


# # Set parameters
# bands = ["B4", "B3", "B2"]

# # date_pattern = "mm_dd_yyyy"  # dd: day, MMM: month (JAN), y: year
# folder = "Tanzania_Fields"


# # extra = dict(sat="Sen_TOA")
# CLOUD_FILTER = 75


# # %% MONTHLY COMPOSITES Hue
# for site, site_name in zip([train_site, test_site], ["train", "test"]):
#     for year in list(range(2017, 2018)):
#         for month in list(range(1, 13, 1)):
#             print(site_name, year, month)
#             dt = pendulum.datetime(year, month, 1)

#             collection = get_s2A_SR_sr_cld_col(
#                 site,
#                 dt.start_of("month").strftime(r"%Y-%m-%d"),
#                 dt.end_of("month").strftime(r"%Y-%m-%d"),
#                 CLOUD_FILTER=CLOUD_FILTER,
#             )
#             s2_sr = (
#                 collection.map(add_cld_shdw_mask)
#                 .map(apply_cld_shdw_mask)
#                 .select(bands)
#                 .median()
#                 .divide(10000)
#                 .rgbToHsv()
#                 .clip(site)
#             )
#             # print(s2_sr.getInfo())

#             hue = s2_sr.select("hue").multiply(100)

#             hue = hue.unmask(9999)

#             hue = geetools.batch.utils.convertDataType("uint16")(hue)

#             # # export clipped result in Tiff

#             img_name = f"SA_{site_name}_S2_SR_hue_M_{year}_{str(month).zfill(2)}"

#             export_config = {
#                 "scale": scale,
#                 "maxPixels": 9000000000,
#                 "driveFolder": folder,
#                 "region": site,
#                 "crs": crs,
#             }
#             task = ee.batch.Export.image(hue, img_name, export_config)
#             task.start()

# # %%

# %%
