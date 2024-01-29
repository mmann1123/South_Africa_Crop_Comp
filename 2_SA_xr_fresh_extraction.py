# %% env xr_fresh

import xarray as xr
import geowombat as gw
import os, sys

sys.path.append("/home/mmann1123/Documents/github/xr_fresh/")
# from xr_fresh.feature_calculators import *
# from xr_fresh.backends import Cluster
# from xr_fresh.extractors import extract_features
from glob import glob
from datetime import datetime
import matplotlib.pyplot as plt

# from xr_fresh.utils import *
import logging
import xarray as xr
from numpy import where

# from xr_fresh import feature_calculators
from itertools import chain
from geowombat.backends import concat as gw_concat

_logger = logging.getLogger(__name__)
from numpy import where
from pathlib import Path
import time
import re

# # start cluster
# cluster = Cluster()
# cluster.start_large_object()

missing_data = 9999


complete_f = {
    "minimum": [{}],
    "abs_energy": [{}],
    "mean_abs_change": [{}],
    "variance_larger_than_standard_deviation": [{}],
    # NOT USEFUL "ratio_beyond_r_sigma": [{"r": 1}, {"r": 2}, {"r": 3}],
    "symmetry_looking": [{}],
    "sum_values": [{}],
    # didn't get selected "autocorr": [
    # {"lag": 1},
    # {"lag": 2},
    # ],  #  not possible in 2023{"lag": 4}],
    "ts_complexity_cid_ce": [{}],
    "mean_change": [{}],  #  FIX  DONT HAVE
    "mean_second_derivative_central": [{}],
    "median": [{}],
    "mean": [{}],
    "standard_deviation": [{}],
    "variance": [{}],
    "skewness": [{}],
    "kurtosis": [{}],
    "absolute_sum_of_changes": [{}],
    "longest_strike_below_mean": [{}],
    # not selected "longest_strike_above_mean": [{}],
    # not selected "count_above_mean": [{}],
    "count_below_mean": [{}],
    # not selected "doy_of_maximum_first": [
    #     {"band": band_name}
    # ],  # figure out how to remove arg for band
    # "doy_of_minimum_first": [{"band": band_name}],
    "doy_of_maximum_first": [{"band": band_name}],
    "ratio_value_number_to_time_series_length": [{}],
    "quantile": [{"q": 0.05}, {"q": 0.95}],
    "maximum": [{}],
    "linear_time_trend": [{"param": "all"}],
}

sys.path.append("/home/mmann1123/Documents/github/xr_fresh")

import xr_fresh as xf

from xr_fresh.feature_calculator_series import doy_of_maximum, abs_energy, maximum, mean


# %%
for band_name in ["B12"]:  # "B11", "B12" "B2",  "B6", "EVI", 'hue'
    files = f"/home/mmann1123/extra_space/Dropbox/Tanzania_data/Projects/YM_Tanzania_Field_Boundaries/Land_Cover/northern_TZ_data/{band_name}"
    file_glob = f"{files}/*.tif"

    f_list = sorted(glob(file_glob))
    f_list

    # Get unique grid codes
    pattern = r"(?<=-)\d+-\d+(?=\.tif)"
    unique_grids = list(
        set(
            [
                re.search(pattern, file_path).group()
                for file_path in f_list
                if re.search(pattern, file_path)
            ]
        )
    )

    # Print the unique codes
    for grid in unique_grids:
        print("working on grid", grid)
        a_grid = sorted([f for f in f_list if grid in f])
        print(a_grid)
        # get dates
        strp_glob = f"{files}/SA_{band_name}_1C_%Y_%m.tif"
        dates = [datetime.strptime(string, strp_glob) for string in a_grid]
        print(dates)

        # # add data notes
        # Path(f"{files}/annual_features").mkdir(parents=False, exist_ok=True)
        # with open(f"{files}/annual_features/0_notes.txt", "a") as the_file:
        #     the_file.write(
        #         "Gererated by /home/mmann1123/Documents/github/YM_TZ_crop_classifier/2_xr_fresh_extraction.py \t"
        #     )
        #     the_file.write(str(datetime.now()))

        with gw.series(
            a_grid,
            nodata=missing_data,
        ) as src:
            print(src)
            src.apply(
                func=abs_energy(),
                outfile=f"/home/mmann1123/Downloads/{band_name}_{grid}_abs_energy.tif",
                num_workers=5,
                bands=1,
            )

# %%
# %%
# # open xarray lazy
# with gw.open(
#     a_grid,
#     band_names=[band_name],
#     nodata=missing_data,
#     time_names=dates,
# ) as ds:
#     ds = ds.chunk(
#         {"time": -1, "band": 1, "y": 350, "x": 350}
#     )  # rechunk to time
#     ds = ds.gw.mask_nodata()
#     print(ds)

#     # generate features current March - Aug ( Msimu growing season)

#     for year in [2023]:
#         year = str(year)
#         print(year)
#         ds_year = ds.sel(time=slice(year + "-03-01", year + "-07-01"))
#         print("interpolating")
#         ds_year = ds_year.interpolate_na(dim="time", limit=4)

#         # make output folder
#         outpath = os.path.join(files, "annual_features/Mar_Aug_S2")
#         os.makedirs(outpath, exist_ok=True)

#         # extract growing season year month day
#         features = extract_features(
#             xr_data=ds_year,
#             feature_dict=complete_f,
#             band=band_name,
#             na_rm=True,
#             persist=True,
#             filepath=outpath,
#             postfix="_mar_aug_" + year,
#         )

# # %%

# %%   yaswanth
for band_name in ["B12"]:  # "B11", "B12" "B2",  "B6", "EVI", 'hue'
    files = f"/home/mmann1123/extra_space/Dropbox/Tanzania_data/Projects/YM_Tanzania_Field_Boundaries/Land_Cover/northern_TZ_data/{band_name}"
    file_glob = f"{files}/*.tif"

    f_list = sorted(glob(file_glob))
    f_list

    # Get unique grid codes
    pattern = r"(?<=-)\d+-\d+(?=\.tif)"
    unique_grids = list(
        set(
            [
                re.search(pattern, file_path).group()
                for file_path in f_list
                if re.search(pattern, file_path)
            ]
        )
    )

    # Print the unique codes
    for grid in unique_grids:
        print("working on grid", grid)
        a_grid = sorted([f for f in f_list if grid in f])
        print(a_grid)
        # get dates
        strp_glob = f"{files}/S2_SR_{band_name}_M_%Y_%m-{grid}.tif"
        dates = [datetime.strptime(string, strp_glob) for string in a_grid]
        print(dates)

        # # add data notes
        # Path(f"{files}/annual_features").mkdir(parents=False, exist_ok=True)
        # with open(f"{files}/annual_features/0_notes.txt", "a") as the_file:
        #     the_file.write(
        #         "Gererated by /home/mmann1123/Documents/github/YM_TZ_crop_classifier/2_xr_fresh_extraction.py \t"
        #     )
        #     the_file.write(str(datetime.now()))

        # open xarray lazy
        with gw.open(
            a_grid,
            band_names=[band_name],
            nodata=missing_data,
            time_names=dates,
        ) as ds:
            ds = ds.chunk(
                {"time": -1, "band": 1, "y": 350, "x": 350}
            )  # rechunk to time
            ds = ds.gw.mask_nodata()
            print(ds)

            # generate features current March - Aug ( Msimu growing season)

            for year in [2017]:
                year = str(year)
                print(year)
                ds_year = ds.sel(time=slice(year + "-01-01", year + "-12-31"))
                print("interpolating")
                ds_year = ds_year.interpolate_na(dim="time", limit=4)

                # make output folder
                outpath = os.path.join(files, "annual_features/Mar_Aug_S2")
                os.makedirs(outpath, exist_ok=True)

                # extract growing season year month day
                features = extract_features(
                    xr_data=ds_year,
                    feature_dict=complete_f,
                    band=band_name,
                    na_rm=True,
                    persist=True,
                    filepath=outpath,
                    postfix="_jan_dec_" + year,
                )

# %%

# for band_name in ["hue"]:  # "B11", "B12" "B2",  "B6", "EVI", 'hue'
#     files = f"/home/mmann1123/extra_space/Dropbox/Tanzania_data/Projects/YM_Tanzania_Field_Boundaries/Land_Cover/northern_TZ_data/{band_name}"
#     file_glob = f"{files}/*.tif"
#     strp_glob = f"{files}/S2_SR_{band_name}_M_%Y_%m-*.tif"

#     f_list = sorted(glob(file_glob))
#     print(f_list)

#     dates = sorted(datetime.strptime(string, strp_glob) for string in f_list)
#     print(dates)

#     # add data notes
#     Path(f"{files}/annual_features").mkdir(parents=False, exist_ok=True)
#     with open(f"{files}/annual_features/0_notes.txt", "a") as the_file:
#         the_file.write(
#             "Gererated by /home/mmann1123/Documents/github/YM_TZ_crop_classifier/2_xr_fresh_extraction.py \t"
#         )
#         the_file.write(str(datetime.now()))

#     # open xarray lazy
#     with gw.open(
#         sorted(glob(file_glob)),
#         band_names=[band_name],
#         time_names=dates,
#         nodata=missing_data,
#     ) as ds:
#         ds = ds.chunk({"time": -1, "band": 1, "y": 350, "x": 350})  # rechunk to time
#         ds = ds.gw.mask_nodata()
#         # ds.attrs["nodatavals"] = (0,)
#         print(ds)

#         # generate features current March - Aug ( Msimu growing season)

#         for year in [2023]:
#             year = str(year)
#             print(year)
#             ds_year = ds.sel(time=slice(year + "-03-01", year + "-06-01"))
#             print("interpolating")
#             ds_year = ds_year.interpolate_na(dim="time", limit=3)

#             # make output folder
#             outpath = os.path.join(files, "annual_features/Mar_Aug_S2")
#             os.makedirs(outpath, exist_ok=True)

#             # extract growing season year month day
#             features = extract_features(
#                 xr_data=ds_year,
#                 feature_dict=complete_f,
#                 band=band_name,
#                 na_rm=True,
#                 persist=True,
#                 filepath=outpath,
#                 postfix="_mar_aug_" + year,
#             )

# # generate features previous Sep - current March ( Masika growing season)

# for year in [2023]:
#     previous_year = str(year - 1)
#     year = str(year)
#     print(year)
#     ds_year = ds.sel(time=slice(previous_year + "-08-01", year + "-03-01"))
#     print("interpolating")
#     ds_year = ds_year.interpolate_na(dim="time", limit=5)

#     # make output folder
#     outpath = os.path.join(files, "annual_features/Sep_Mar_S2")
#     os.makedirs(outpath, exist_ok=True)

#     # extract growing season year month day
#     features = extract_features(
#         xr_data=ds_year,
#         feature_dict=complete_f,
#         band=band_name,
#         na_rm=True,
#         persist=True,
#         filepath=outpath,
#         postfix="_sep_mar_" + year,
#     )
# %% change name for quantiles to remove . from name
from glob import glob
import os

os.chdir(
    "/home/mmann1123/extra_space/Dropbox/Tanzania_data/Projects/YM_Tanzania_Field_Boundaries/Land_Cover"
)
files = glob("./data/**/annual_features/**/*_quantile_*.tif")

# for files update name to remove 0.05 and 0.95 with 0_05  from file name
for file in files:
    new_file = file.replace("0.95", "0_95")
    new_file = new_file.replace("0.05", "0_05")
    os.rename(file, new_file)
# %%
