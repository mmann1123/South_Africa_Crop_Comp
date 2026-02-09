# %% FEATURE EXTRACTION USING SERIES
########################################################

from xr_fresh.feature_calculator_series import *
from xr_fresh.feature_calculator_series import function_mapping
import geowombat as gw
from glob import glob
from datetime import datetime
import os
import numpy as np
import logging

os.chdir(r"C:\Users\mmann1123\Dropbox\South_Africa_data\Projects\Agriculture_Comp\S1c_data")
 
# Set up logging
logging.basicConfig(
    filename="../features/error_log.log",
    level=logging.ERROR,
    format="%(asctime)s:%(levelname)s:%(message)s",
)
complete_times_series_list = {
    "abs_energy": [{}],
    "absolute_sum_of_changes": [{}],
    "autocorr": [{"lag": 1}, {"lag": 2}, {"lag": 3}],
    "count_above_mean": [{}],
    "count_below_mean": [{}],
    "doy_of_maximum": [{}],
    "doy_of_minimum": [{}],
    "kurtosis": [{}],
    "large_standard_deviation": [{}],
    # # # "longest_strike_above_mean": [{}],  # not working with jax GPU ram issue
    # # # "longest_strike_below_mean": [{}],  # not working with jax GPU ram issue
    "maximum": [{}],
    "mean": [{}],
    "mean_abs_change": [{}],
    "mean_change": [{}],
    "mean_second_derivative_central": [{}],
    "median": [{}],
    "minimum": [{}],
    # "ols_slope_intercept": [
    #     {"returns": "intercept"},
    #     {"returns": "slope"},
    #     {"returns": "rsquared"},
    # ],  # not working
    "quantile": [{"q": 0.05}, {"q": 0.95}],
    "ratio_beyond_r_sigma": [{"r": 1}, {"r": 2}],
    "skewness": [{}],
    "standard_deviation": [{}],
    "sum": [{}],
    "symmetry_looking": [{}],
    "ts_complexity_cid_ce": [{}],
    "variance": [{}],
    "variance_larger_than_standard_deviation": [{}],
}


for band_name in ["B12", "B11", "B2", "B6", "EVI", "hue"]:
    file_glob = f"**/*{band_name}*.tif"

    f_list = sorted(glob(file_glob))
     
    # update doy with dates
    date_pattern =        f"{band_name}\\SA_{band_name}_1C_%Y_%m.tif"
    dates =[datetime.strptime(i,date_pattern) for i in f_list]
    complete_times_series_list["doy_of_maximum"] = [{"dates": dates}]
    complete_times_series_list["doy_of_minimum"] = [{"dates": dates}]

    with gw.series(
        f_list,
        window_size=[512, 512],  # transfer_lib="numpy"
        nodata=np.nan,
    ) as src:
        # iterate across functions
        for func_name, param_list in complete_times_series_list.items():
            for params in param_list:
                # instantiate function
                func_class = function_mapping.get(func_name)
                if func_class:
                    func_instance = func_class(
                        **params
                    )  # Instantiate with parameters
                    if len(params) > 0:
                        print(f"Instantiated {func_name} with  {params}")
                    else:
                        print(f"Instantiated {func_name} ")

                # create output file name
                if len(list(params.keys())) > 0:
                    key_names = list(params.keys())[0]
                    value_names = list(params.values())[0]
                    outfile = f"../features/{band_name}_{func_name}_{key_names}_{value_names}.tif"
                    # avoid issue with all dates
                    if func_name in ["doy_of_maximum", "doy_of_minimum"]:
                        outfile = f"../features/{band_name}_{func_name}_{key_names}.tif"
                else:
                    outfile = f"../features/{band_name}_{func_name}.tif"
                # extract features
                try:
                    src.apply(
                        func=func_instance,
                        outfile=outfile,
                        num_workers=3,
                        processes=False,
                        bands=1,
                        kwargs={"BIGTIFF": "YES", "compress": "LZW"},
                    )
                except Exception as e:
                    logging.error(
                        f"Error extracting features from {band_name} {func_name}: {e}"
                    )
                    continue

# %%
