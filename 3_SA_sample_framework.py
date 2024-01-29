

#%%
# Use geowombat to extract imagery to polygons
from glob import glob
import geopandas as gpd
import os
import geowombat as gw
import numpy as np

polys = glob(r'C:\Users\mmann1123\Dropbox\South_Africa_data\Projects\Agriculture_Comp\ref_fusion_competition_south_africa_train_labels/**/*.geojson')
polys

os.chdir(r"C:\Users\mmann1123\Dropbox\South_Africa_data\Projects\Agriculture_Comp\features")


for poly, poly_label in zip(polys,['data_34S_19E_258N','data_34S_19E_259N']):
    for band_name in ["B12", "B11", "B2", "B6", "EVI", "hue"]:
        file_glob = f"*{band_name}*.tif"
        f_list = sorted(glob(file_glob))

        with gw.open(f_list,band_names=[i.split('.')[0] for i in f_list],stack_dim='band',
                    nodata=np.nan) as src:
            display(src)
            df = gw.extract(src, poly, all_touched =True, )
            df.to_parquet(f'./{band_name}_{poly_label}.parquet', 
                          engine='auto', 
                          compression='snappy')
# %%
