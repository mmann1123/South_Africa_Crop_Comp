# %% env xr_fresh_update on desktop

# Purpose: Extract features from raster data to polygons

import geowombat as gw
from geowombat.core.parallel import ParallelTask
import geopandas as gpd
import rasterio as rio
import ray
from ray.util import ActorPool
from glob import glob
import os
import numpy as np
import pandas as pd


@ray.remote
class Actor(object):
    def __init__(self, aoi_id=None, id_column=None, band_names=None):
        self.aoi_id = aoi_id
        self.id_column = id_column
        self.band_names = band_names

    # While the names can differ, these three arguments are required.
    # For ``ParallelTask``, the callable function within an ``Actor`` must be named exec_task.
    def exec_task(self, data_block_id, data_slice, window_id):
        data_block = data_block_id[data_slice]
        left, bottom, right, top = data_block.gw.bounds
        aoi_sub = self.aoi_id.cx[left:right, bottom:top]

        if aoi_sub.empty:
            return aoi_sub

        # Return a GeoDataFrame for each actor
        return gw.extract(
            data_block, aoi_sub, id_column=self.id_column, band_names=self.band_names
        )


# %%
################################################################
# extract time series features for all bands
################################################################

# import other necessary modules...
os.chdir(
    r"C:\Users\mmann1123\Dropbox\South_Africa_data\Projects\Agriculture_Comp\features"
)

polys = glob(
    r"C:\Users\mmann1123\Dropbox\South_Africa_data\Projects\Agriculture_Comp\ref_fusion_competition_south_africa_train_labels/**/*.geojson"
)
polys


ray.init(local_mode=True)  # this avoids thread conflicts for large objects

for band_name in ["B12", "B11", "B2", "B6", "EVI", "hue"]:
    for poly_i, poly_label in zip([0, 1], ["34S_19E_258N", "34S_19E_259N"]):
        with rio.Env(GDAL_CACHEMAX=256 * 1e6) as env:
            file_glob = f"*{band_name}*.tif"
            f_list = sorted(glob(file_glob))
            df_id = ray.put(gpd.read_file(polys[poly_i]).to_crs("EPSG:4326"))

            band_names = [i.split(".ti")[0] for i in f_list]

            # Since we are iterating over the image block by block, we do not need to load
            # a lazy dask array (i.e., chunked).
            with gw.open(
                f_list, band_names=band_names, stack_dim="band", chunks=16
            ) as src:

                # Setup the pool of actors, one for each resource available to ``ray``.
                actor_pool = ActorPool(
                    [
                        Actor.remote(
                            aoi_id=df_id, id_column="id", band_names=band_names
                        )
                        for n in range(0, int(ray.cluster_resources()["CPU"]))
                    ]
                )

                # Setup the task object
                pt = ParallelTask(
                    src,
                    row_chunks=4096,
                    col_chunks=4096,
                    scheduler="ray",
                    # n_chunks=1000,
                )
                results = pt.map(actor_pool)

        del df_id, actor_pool
        ray.shutdown()
        results2 = [df.reset_index(drop=True) for df in results if len(df) > 0]
        result = pd.concat(results2, ignore_index=True, axis=0)
        result = pd.DataFrame(result.drop(columns="geometry"))
        result.to_parquet(
            f"./{band_name}_{poly_label}.parquet", engine="auto", compression="snappy"
        )

# %%
################################################################
# extract raw bands
################################################################

# import other necessary modules...
os.chdir(r"/mnt/bigdrive/Dropbox/South_Africa_data/Projects/Agriculture_Comp/S1c_data/")

polys = glob(
    r"/mnt/bigdrive/Dropbox/South_Africa_data/Projects/Agriculture_Comp/ref_fusion_competition_south_africa_train_labels/**/*.geojson"
)

polys
# %%  Local mode seems to avoid thread conflicts for large objects
ray.init(local_mode=True)

for band_name in ["B12", "B11", "B2", "B6", "EVI", "hue"]:
    for poly_i, poly_label in zip([0, 1], ["34S_19E_258N", "34S_19E_259N"]):

        with rio.Env(GDAL_CACHEMAX=256 * 1e6) as env:

            f_list = sorted(glob(f"*{band_name}*/*.tif"))
            df_id = ray.put(gpd.read_file(polys[poly_i]).to_crs("EPSG:4326"))

            band_names = [i.split(".ti")[0] for i in f_list]

            with gw.open(
                f_list,
                band_names=band_names,
                stack_dim="band",
                chunks=2048,  # chunks controls how much is read in at once 1 is largest
            ) as src:
                display(src)
                chunk_sizes = src.chunks
                print(chunk_sizes)

                actor_pool = ActorPool(
                    [
                        Actor.remote(
                            aoi_id=df_id, id_column="id", band_names=band_names
                        )
                        for n in range(0, int(ray.cluster_resources()["CPU"]))
                    ]
                )

                pt = ParallelTask(
                    src,
                    row_chunks=2048,
                    col_chunks=2048,
                    scheduler="ray",
                )
                results = pt.map(actor_pool)

        del df_id, actor_pool, pt
        results2 = [df.reset_index(drop=True) for df in results if len(df) > 0]
        result = pd.concat(results2, ignore_index=True, axis=0)
        result = pd.DataFrame(result.drop(columns="geometry"))
        result.to_parquet(
            f"./{band_name}_raw_{poly_label}.parquet",
            engine="auto",
            compression="snappy",
        )

# %% View the results
import pandas as pd

data = pd.read_parquet(
    "/mnt/bigdrive/Dropbox/South_Africa_data/Projects/Agriculture_Comp/S1c_data/B2_raw_34S_19E_258N.parquet"
)
data

# %%
# # %%  Experiment with local mode

# import ray

# # Initialize Ray in local mode
# ray.init(local_mode=True)

# import cProfile
# import pstats
# import io


# def process_bands():
#     for band_name in ["B12"]:  # , "B11", "B2", "B6", "EVI", "hue"]:
#         for poly_i, poly_label in zip([0, 1], ["34S_19E_258N", "34S_19E_259N"]):

#             with rio.Env(GDAL_CACHEMAX=256 * 1e6) as env:

#                 f_list = sorted(glob(f"*{band_name}*/*.tif"))[0:1]
#                 df_id = ray.put(gpd.read_file(polys[poly_i]).to_crs("EPSG:4326"))

#                 band_names = [i.split(".ti")[0] for i in f_list]

#                 with gw.open(
#                     f_list, band_names=band_names, stack_dim="band", chunks=16
#                 ) as src:

#                     actor_pool = ActorPool(
#                         [
#                             Actor.remote(
#                                 aoi_id=df_id, id_column="id", band_names=band_names
#                             )
#                             for n in range(0, int(ray.cluster_resources()["CPU"]))
#                         ]
#                     )

#                     pt = ParallelTask(
#                         src,
#                         row_chunks=2048 // 4,
#                         col_chunks=2048 // 4,
#                         scheduler="ray",
#                     )
#                     results = pt.map(actor_pool)

#             del df_id, actor_pool, pt
#             results2 = [df.reset_index(drop=True) for df in results if len(df) > 0]
#             result = pd.concat(results2, ignore_index=True, axis=0)
#             result = pd.DataFrame(result.drop(columns="geometry"))
#             result.to_parquet(
#                 f"./{band_name}_raw_{poly_label}.parquet",
#                 engine="auto",
#                 compression="snappy",
#             )


# # Profile the function
# pr = cProfile.Profile()
# pr.enable()
# process_bands()
# pr.disable()

# # Print profiling results
# s = io.StringIO()
# sortby = "cumulative"
# ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
# ps.print_stats()
# print(s.getvalue())
# # %%
# # write to text file
# with open("profile_stats.txt", "w") as f:
#     f.write(s.getvalue())

# %% TOO SLOW

# # Use geowombat to extract imagery to polygons
# from glob import glob
# import geopandas as gpd
# import os
# import geowombat as gw
# import numpy as np

# polys = glob(
#     r"C:\Users\mmann1123\Dropbox\South_Africa_data\Projects\Agriculture_Comp\ref_fusion_competition_south_africa_train_labels/**/*.geojson"
# )
# polys

# os.chdir(
#     r"C:\Users\mmann1123\Dropbox\South_Africa_data\Projects\Agriculture_Comp\features"
# )

# # %%
# for poly, poly_label in zip(polys, ["data_34S_19E_258N", "data_34S_19E_259N"]):
#     for band_name in ["B12", "B11", "B2", "B6", "EVI", "hue"]:
#         print(f"working on {band_name} {poly_label}")
#         file_glob = f"*{band_name}*.tif"
#         f_list = sorted(glob(file_glob))

#         with gw.open(
#             f_list,
#             band_names=[i.split(".")[0] for i in f_list],
#             stack_dim="band",
#             nodata=np.nan,
#         ) as src:
#             display(src)
#             df = gw.extract(
#                 src,
#                 poly,
#                 all_touched=True,
#             )
#             print(df.head())
#             df.to_parquet(
#                 f"./{band_name}_{poly_label}.parquet",
#                 engine="auto",
#                 compression="snappy",
#             )
# # %%
# import cProfile
# import numpy as np

# # import other necessary modules...


# def main():

#     with gw.open(
#         f_list,
#         band_names=[i.split(".")[0] for i in f_list],
#         stack_dim="band",
#         nodata=np.nan,
#     ) as src:
#         display(src)
#         df = gw.extract(src, poly, all_touched=True)
#         print(df.head())
#         df.to_parquet(
#             f"./{band_name}_{poly_label}.parquet", engine="auto", compression="snappy"
#         )


# cProfile.run("main()", "profile_stats")
