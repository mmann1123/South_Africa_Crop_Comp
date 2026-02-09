##############################################################
# CLASS DIFFERENTIATION CHECK
###############################################################

# %%  Figure out how to reassign similar classes
import geopandas as gpd
from sklearn.decomposition import IncrementalPCA, PCA
import geowombat as gw
from glob import glob
import os
from dask.distributed import Client, LocalCluster

os.chdir(
    "/home/mmann1123/extra_space/Dropbox/Tanzania_data/Projects/YM_Tanzania_Field_Boundaries/Land_Cover/"
)

missing_data = 9999

# Get all the feature files
files = sorted(glob("./data/**/annual_features/**/**.tif"))
# Get the names of the bands
band_names = [os.path.basename(f).split(".")[0] for f in files]
# Read the training data
data = gpd.read_file("./data/training_cleaned.geojson")
data["Primary land cover"].unique()
# %%

# Only keep the land cover and geometry columns
data = data[["Primary land cover", "geometry"]]
# restrict land cover classes

# 'vegetables','other', 'speciality_crops', 'eggplant',  'tree_crops', '', 'okra', '', 'don_t_know'

keep = [
    "cassava",
    "maize",
    "rice",
    "cotton",
    "sorghum",
    "millet",
    "soybeans",
    "sunflower",
    "other_grain",
]
data.loc[data["Primary land cover"].isin(keep) == False, "Primary land cover"] = "Other"
data["lc_name"] = data["Primary land cover"]

# add additional training data
other_training = gpd.read_file("./data/other_training.gpkg").to_crs(data.crs)

lu_complete = data[["lc_name", "geometry"]].overlay(
    other_training[["lc_name", "geometry"]], how="union"
)
lu_complete["lc_name"] = lu_complete["lc_name_1"].fillna(lu_complete["lc_name_2"])

lu_complete["lc_name"]


# %% GET IMAGES EXTRACTED

target_string = next((string for string in files if "EVI" in string), None)

with gw.config.update(ref_image=target_string):
    # open the data using geowombat.open()
    with gw.open(
        files,
        stack_dim="band",
        band_names=band_names,
        nodata=missing_data,
        resampling="nearest",
    ) as src:
        # use geowombat.extract() to extract data
        X = gw.extract(
            src,
            lu_complete,
            all_touched=True,
        )
        print(X)

# %% Calc cluster assignments

from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Create a dictionary to store key names for each target_value
result_dict = {}


for state in range(0, 30):
    # Initialize a LabelEncoder
    le = LabelEncoder()

    # Initialize a pipeline with a variance thresholding, data imputation, standard scaling, and K-means steps
    pipeline = Pipeline(
        [
            # ("variance_threshold", VarianceThreshold()),
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            (
                "kmeans",
                KMeans(
                    n_clusters=int(len(lu_complete["lc_name"].unique())),
                    random_state=state,
                ),
            ),
        ]
    )

    # Fit the pipeline on your training data
    Xtrans = pipeline.fit_transform(X.values[:, 4:])
    Xtrans.shape

    # recode y values to integers
    y = le.fit_transform(X["lc_name"])

    cluster_labels = pipeline["kmeans"].labels_
    class_labels = np.unique(y)
    class_names = le.inverse_transform(class_labels)

    # Create a dictionary to store the mapping of original class labels to cluster labels
    class_to_cluster = {}

    # Iterate through each original class and find the corresponding cluster label
    for class_label in class_labels:
        cluster_label = np.argmax(np.bincount(cluster_labels[y == class_label]))
        class_to_cluster[le.inverse_transform([class_label])[0]] = cluster_label
    class_to_cluster

    # Iterate through unique target_values
    for key, target_value in class_to_cluster.items():
        # Collect key names with the same target_value
        if state > 0:
            temp_dict = {}
            temp_dict[key] = [
                key for key, value in class_to_cluster.items() if value == target_value
            ]
            result_dict[key] = result_dict[key] + temp_dict[key]
        else:
            result_dict[key] = [
                key for key, value in class_to_cluster.items() if value == target_value
            ]
print(result_dict)


# %%  VISUALIZE

# Convert the dictionary to a pandas DataFrame
data2 = []
for key, values in result_dict.items():
    data2.extend([(key, value) for value in values])
df = pd.DataFrame(data2, columns=["Key", "Value"])

# Create a FacetGrid with histograms
g = sns.FacetGrid(df, col="Key")
g.map(sns.histplot, "Value", bins=len(set(df["Value"])), kde=False)

# Set labels and title
g.set_axis_labels("Value", "Count")
g.fig.suptitle("Value Counts by Key")
g.set_xticklabels(rotation=90)

# Adjust the spacing between subplots
g.tight_layout()

# Display the plot
plt.show()

# save
g.savefig("./outputs/cluster_reassignment.png")
# %%


# %%

import geowombat as gw
from glob import glob
from datetime import datetime
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
from datetime import datetime

files = "/home/mmann1123/extra_space/Dropbox/Tanzania_data/Projects/YM_Tanzania_Field_Boundaries/Land_Cover/data"
band_name = "EVI"
file_glob = f"{files}/{band_name}/*.tif"
strp_glob = f"{files}/{band_name}/S2_SR_EVI_M_%Y_%M.tif"

f_list = sorted(glob(file_glob))
dates = [
    "EVI-" + datetime.strptime(string, strp_glob).strftime("%M%Y") for string in f_list
]
print(f_list)
dates
# %%
points = gpd.read_file(
    "/home/mmann1123/extra_space/Dropbox/Tanzania_data/Projects/YM_Tanzania_Field_Boundaries/kobo_field_collections/TZ_ground_truth_cleaned.gpkg"
)

with gw.open(file_glob, nodata=0, stack_dim="time") as src:
    df = src.gw.extract(points, time_names=dates)
    df.replace(0, np.nan, inplace=True)
    display(df)

# %%
dates = [string + "_1" for string in dates]
dates.append("Primary land cover")
# %%
df2 = df[dates]
df2["id"] = df2.index
df2
# %%
import pandas as pd

df2 = pd.wide_to_long(
    df2, stubnames=["EVI"], i="id", j="yearmonth", sep="-", suffix=r"\w+"
)
df2.reset_index(inplace=True)
# %%
from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()
df2["LC_code"] = lb.fit_transform(df2["Primary land cover"])
df2
# %%
import seaborn as sns

g = sns.scatterplot(
    data=df2.groupby(["Primary land cover", "yearmonth"], as_index=False).mean(),
    x="yearmonth",
    y="EVI",
    hue="Primary land cover",
    # row="Primary land cover",
    legend=False,
)
g.set_xticklabels(labels=df2.yearmonth.unique(), rotation=90)

# %%
g = sns.FacetGrid(
    df2.groupby(["Primary land cover", "yearmonth"], as_index=False).median(),
    col="Primary land cover",
    col_wrap=1,
    hue="Primary land cover",
)
g = g.map(plt.scatter, "yearmonth", "EVI")

# %%
sns.lmplot(
    data=df2.groupby(["LC_code", "yearmonth"], as_index=False).mean(),
    x="yearmonth",
    y="EVI",
    hue="LC_code",
    lowess=True,
)
# %%

g = sns.FacetGrid(
    df2.groupby(["Primary land cover", "yearmonth"], as_index=False).mean(),
    col="Primary land cover",
    col_wrap=2,
)
g = g.map(sns.lmplot, x="yearmonth", y="EVI")
# %%
