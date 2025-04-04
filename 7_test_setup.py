# %%
import pandas as pd
from glob import glob
import os

os.chdir(r"/mnt/bigdrive/Dropbox/South_Africa_data/Projects/Agriculture_Comp/features")
files = glob("testing*.parquet")
files
# %% remove crop_name and id from parquet files for out of sample preds

for file in files:
    data = pd.read_parquet(file).drop(columns=["crop_name", "crop_id"])
    data.to_parquet("X_" + file)

#%% create geofield boundaries without crops 
import geopandas as gpd
polys = glob(
    r"/mnt/bigdrive/Dropbox/South_Africa_data/Projects/Agriculture_Comp/ref_fusion_competition_south_africa_test_labels/ref_fusion_competition_south_africa_test_labels_34S_20E_259N/*.geojson"
)
gpd.read_file(polys[0]).drop(columns=['crop_id','crop_name']).to_file('X_testing_34S_20E_259N.geojson',driver='GeoJSON')

# %% Create answer

pd.read_parquet(files[0]).groupby("id").agg("first")["crop_name"].to_csv('ground_truth.csv',index=False)
!base64 ground_truth.csv

#Go to your GitHub repo → Settings → Secrets and variables → Actions → New repository secret.
# Name it something like GROUND_TRUTH, and paste in your base64-encoded CSV content. Let me know if you want a quick bash command to generate that base64!

# %%
pd.read_parquet(files[0]).groupby("id").agg("first")["crop_name"].unique()
# %%
