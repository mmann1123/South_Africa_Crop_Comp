# %% env:crop_class


# NOTE: ADD REGIONAL DUMMY

# %%
# change directory first
os.chdir(
    "/home/mmann1123/extra_space/Dropbox/Tanzania_data/Projects/YM_Tanzania_Field_Boundaries/Land_Cover"
)
from sklearn_helpers import (
    best_classifier_pipe,
    get_selected_ranked_images,
    classifier_objective,
    extract_top_from_shaps,
    remove_collinear_features,
    remove_list_from_list,
)
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    KFold,
    cross_val_score,
    RandomizedSearchCV,
    StratifiedKFold,
    StratifiedGroupKFold,
)

from sklearn.metrics import (
    confusion_matrix,
    cohen_kappa_score,
    accuracy_score,
)
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    LabelEncoder,
    OneHotEncoder,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans, OPTICS
import lightgbm as lgb
import xgboost
import optuna
import shap
import sqlite3
from sklearn.metrics import log_loss, balanced_accuracy_score
import os
import geowombat as gw
from geowombat.ml import fit_predict, predict, fit
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import VarianceThreshold
import umap
from glob import glob

# how many images will be selected for importances
select_how_many = 15


from glob import glob
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd


# remove nan and bad columns
pipeline_scale_clean = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("variance_threshold", VarianceThreshold(threshold=0.5)),
    ]
)
pipeline_scale = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ]
)

# %%
# read YM training data and clean
import geopandas as gpd

lu = gpd.read_file("./data/training_cleaned.geojson")


# get buffer size based on field size
lu.Field_size.replace(
    {"small": 10, "medium": 20, "large": 30, np.nan: 10}, inplace=True
)


np.unique(lu["Primary land cover"])

# restrict land cover classes

# order of importance to USDA
# Corn  (technically Masika Corn, as there are 3 crops over the year)
# Cotton
# Rice
# Sorghum
# Millet
# Other grains (wheat, barley, oats, rye…)
# Sunflower
# Cassava
# Soybeans

# print # of obs per class
print(lu["Primary land cover"].value_counts())


lu["lc_name"] = lu["Primary land cover"]
keep = [
    "rice",
    "maize",
    "cassava",
    # "vegetables",
    "sunflower",
    "sorghum",
    # "other",
    "cotton",
    # "speciality_crops",
    # "okra ",
    # "eggplant",
    # "soybeans",
    # "tree_crops",
    "millet",
    # "other_grain",
]
drop = [
    "Don't know",
    "Other (later, specify in optional notes)",
]

# apply keep/drop
lu.drop(lu[lu["lc_name"].isin(drop)].index, inplace=True)
lu.loc[lu["lc_name"].isin(keep) == False, "lc_name"] = "Other"

# combine sorghum and millet
lu.lc_name.replace(
    {"millet": "millet_sorghum", "sorghum": "millet_sorghum"}, inplace=True
)

# add additional training data
other_training = gpd.read_file("./data/other_training.gpkg").to_crs(lu.crs)
other_training["Field_size"] = 20


lu_complete = lu[["lc_name", "Field_size", "Quality", "geometry"]].overlay(
    other_training[["lc_name", "Field_size", "geometry"]], how="union"
)
lu_complete["lc_name"] = lu_complete["lc_name_1"].fillna(lu_complete["lc_name_2"])
lu_complete["Field_size"] = lu_complete["Field_size_1"].fillna(
    lu_complete["Field_size_2"]
)


# drop two missing values
lu_complete.dropna(subset=["lc_name"], inplace=True)
# fill missing quality for other training data
lu_complete["Quality"].fillna("OK", inplace=True)


# The labels are string names, so here we convert them to integers
le = LabelEncoder()
lu_complete["lc"] = le.fit_transform(lu_complete["lc_name"])
print(lu_complete["lc"].unique())

# buffer points based on filed size
lu_poly = lu_complete.copy()
lu_poly["geometry"] = lu_poly.apply(lambda x: x.geometry.buffer(x.Field_size), axis=1)


# images = glob("./data/EVI/annual_features/*/**.tif")


# Get all the feature files
images = sorted(glob("./data/**/annual_features/**/**.tif"))
# remove dropbox case conflict images
images = [item for item in images if "(Case Conflict)" not in item]

# drop medium and low quality sites
lu_complete[lu_complete.Quality.str.contains("OK")]


# %%
########################################################
# Get LGBM parameters for feature selection
########################################################
# uses select_how_many from top of script

target_string = next((string for string in images if "EVI" in string), None)
with gw.config.update(ref_image=target_string):
    with gw.open(images, nodata=9999, stack_dim="band") as src:
        # fit a model to get Xy used to train model
        df = gw.extract(src, lu_poly, verbose=1)
        y = df["lc"]
        X = df[range(1, len(images) + 1)]
        X.columns = [os.path.basename(f).split(".")[0] for f in images]
        groups = df.id.values

X = pipeline_scale_clean.fit_transform(X)


#   Create optuna classifier study

# Create an SQLite connection
conn = sqlite3.connect("models/study.db")

# Create a study with SQLite storage
storage = optuna.storages.RDBStorage(url="sqlite:///models/study.db")

# delete any existing study
try:
    study = optuna.load_study(
        study_name="model_selection_feature_selection", storage=storage
    )
    optuna.delete_study(study_name="model_selection_feature_selection", storage=storage)
except:
    pass

# create new study
study = optuna.create_study(
    storage=storage,
    study_name="model_selection_feature_selection",
    direction="maximize",
)


# Optimize the objective function
study.optimize(
    lambda trial: classifier_objective(
        trial, X, y, classifier_override="LGBM", groups=groups
    ),
    n_trials=100,
    n_jobs=3,
)

# Close the SQLite connection
conn.close()

print("Number of finished trials: {}".format(len(study.trials)))

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

# write params to csv
pd.DataFrame(study.trials_dataframe()).to_csv(
    "models/optuna_study_model_selection_model_selection_feature_selection.csv"
)


#  Save results
conn = sqlite3.connect("models/study.db")

study = optuna.load_study(
    storage="sqlite:///models/study.db",
    study_name="model_selection_feature_selection",
)


# Access the top trials
top_trials = study.best_trials

# Iterate over the top trials and print their scores and parameters
for i, trial in enumerate(top_trials):
    print(f"Rank {i+1}: Score = {trial.value}")
    print(f"Parameters: {trial.params}")

# Get the DataFrame of all trials
trials_df = study.trials_dataframe()

# Sort the trials by the objective value in ascending order
sorted_trials = trials_df.sort_values("value", ascending=False)

# Print the ranked listing of trials
print(sorted_trials[["number", "value", "params_classifier"]])


# %%
########################################################
# FEATURE SELECTION
########################################################

# NOTE combining mean,max shaps and kbest is working well

# %% Extract best parameters for LGBM
lgbm_pipe = best_classifier_pipe(
    "models/study.db", "model_selection_feature_selection", "LGBM"
)
params_lgbm_dict = lgbm_pipe["classifier"].get_params()

# %%
# extract data
target_string = next((string for string in images if "EVI" in string), None)

with gw.config.update(ref_image=target_string):
    with gw.open(images, nodata=9999, stack_dim="band") as src:
        # fit a model to get Xy used to train model
        df = gw.extract(src, lu_poly, verbose=1)
        y = df["lc"]
        y.reset_index(drop=True, inplace=True)
        X = df[range(1, len(images) + 1)]
        X_columns = [os.path.basename(f).split(".")[0] for f in images]
        groups = df.id.values


X = pipeline_scale_clean.fit_transform(X)

skf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=7)

feature_importance_list = []
shaps_importance_list = []
for train_index, test_index in skf.split(X, y, groups=groups):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    for metric in ["multi_error", "multi_logloss"]:
        # Train the LightGBM model
        params = params_lgbm_dict.copy()
        params["objective"] = "multiclass"
        params["metric"] = metric
        params["num_classes"] = len(lu_complete["lc"].unique())
        d_train = lgb.Dataset(X_train, label=y_train)
        d_test = lgb.Dataset(X_test, label=y_test, reference=d_train)
        model = lgb.train(
            params,
            d_train,
            10000,
            valid_sets=[d_test],
            early_stopping_rounds=200,
            verbose_eval=1000,
        )

        # SHAP exaplainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        shaps_importance_list.append(shap_values)

        # feature importance
        shap.summary_plot(
            shap_values,
            X,
            feature_names=[x.replace("_", ".") for x in X_columns],
            class_names=le.classes_,
            plot_type="bar",
            max_display=20,
            plot_size=(10, 10),
        )

#  Calculate mean shapes values
# %%
mean_shaps = [
    np.mean(np.abs(elements), axis=0) for elements in zip(*shaps_importance_list)
]
# feature importance
summary = shap.summary_plot(
    mean_shaps,
    X,
    feature_names=[x.replace("_", ".") for x in X_columns],
    class_names=le.classes_,
    plot_type="bar",
    max_display=20,
    plot_size=(10, 10),
    show=False,
)

plt.savefig(f"outputs/mean_shaps_importance_{select_how_many}.png", bbox_inches="tight")


# %% By default the features are ordered using shap_values.abs.mean(0), which is the mean absolute value of
# the SHAP values for each feature.
# This order however places more emphasis on broad average impact, and less on rare but high magnitude impacts.
# If we want to find features with high impacts for individual classes we can instead sort by the max absolute
# value:

max_shaps = [
    np.max(np.abs(elements), axis=0) for elements in zip(*shaps_importance_list)
]
summary = shap.summary_plot(
    max_shaps,
    X,
    feature_names=[x.replace("_", ".") for x in X_columns],
    class_names=le.classes_,
    plot_type="bar",
    max_display=20,
    plot_size=(10, 10),
    show=False,
)
plt.savefig(f"outputs/max_shaps_importance_{select_how_many}.png", bbox_inches="tight")


# %% write out top features from shaps mean and max
# to "./outputs/selected_images_{file_prefix}_{select_how_many}.csv", index=False


# NOTE I don't think this is working properly check
extract_top_from_shaps(
    shaps_list=mean_shaps,
    column_names=X_columns,
    select_how_many=select_how_many,
    remove_containing=None,
    file_prefix="mean",
)
extract_top_from_shaps(
    shaps_list=max_shaps,
    column_names=X_columns,
    select_how_many=select_how_many,
    remove_containing=None,
    file_prefix="max",
)


# %% Get kbest features from sklearn
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif

pl = Pipeline(
    [
        ("variance_threshold", VarianceThreshold()),  # Remove low variance features
        ("impute", SimpleImputer(strategy="constant", fill_value=9999)),
        (
            "feature_selection",
            SelectKBest(k=select_how_many, score_func=f_classif),
        ),  # Select top k features based on ANOVA F-value
        ("clf", RandomForestClassifier()),
    ]
)

gridsearch = GridSearchCV(
    pl,
    cv=KFold(n_splits=3),
    scoring="balanced_accuracy",
    param_grid={"clf__n_estimators": [1000]},
)

with gw.config.update(ref_image=images[-1]):
    with gw.open(images, nodata=9999, stack_dim="band") as src:
        # fit a model to get Xy used to train model
        X = gw.extract(src, lu_complete)
        y = lu_complete["lc"]
        X = X[range(1, len(images) + 1)]
        del src

gridsearch.fit(X=X.values, y=y.values)
print(gridsearch.cv_results_)
print(gridsearch.best_score_)
print(gridsearch.best_params_)
print(
    [
        os.path.basename(images[i])
        for i in gridsearch.best_estimator_.named_steps[
            "feature_selection"
        ].get_support(indices=True)
    ]
)

kbest_images = [
    images[i]
    for i in gridsearch.best_estimator_.named_steps["feature_selection"].get_support(
        indices=True
    )
]
pd.DataFrame({f"top{select_how_many}names": kbest_images}).to_csv(
    f"./outputs/selected_images_kbest_{select_how_many}.csv",
)

# %%
# FIND REDUNDANT FEATURES #
# # %% feature clustering to find redundant features
# # https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/bar.html#Using-feature-clustering
# Read in the list of selected images

# Trying other method
# https://stackoverflow.com/questions/29294983/how-to-calculate-correlation-between-all-columns-and-remove-highly-correlated-on

# NOTE THIS MIGHT NOT BE WORKING PROPERLY CHECK

select_images = list(
    set(
        list(
            pd.read_csv(f"./outputs/selected_images_mean_{select_how_many}.csv")[
                f"top{select_how_many}names"
            ].values
        )
        + list(
            pd.read_csv(f"./outputs/selected_images_max_{select_how_many}.csv")[
                f"top{select_how_many}names"
            ].values
        )
        + list(
            pd.read_csv(f"./outputs/selected_images_kbest_{select_how_many}.csv")[
                f"top{select_how_many}names"
            ].values
        )
    )
)

target_string = next((string for string in images if "EVI" in string), None)

with gw.config.update(ref_image=target_string):
    with gw.open(select_images, nodata=9999, stack_dim="band") as src:
        # fit a model to get Xy used to train model
        df = gw.extract(src, lu_poly, verbose=1)
        y = df["lc"]
        y.reset_index(drop=True, inplace=True)
        X = df[range(1, len(select_images) + 1)]
        X_columns = [os.path.basename(f).split(".")[0] for f in select_images]
        groups = df.id.values

# remove nan and bad columns

X = pipeline_scale.fit_transform(X)

remove_collinear_features(
    pd.DataFrame(X, columns=X_columns),
    threshold=0.9,
    out_df=f"./outputs/collinear_features_{select_how_many}.csv",
)


##############################################################
# %% RESAMPLE IMAGES
##############################################################
# resample all selected features to 10m and set smallest dtype possible
# Read in the list of selected images
select_images = list(
    set(
        list(
            pd.read_csv(f"./outputs/selected_images_mean_{select_how_many}.csv")[
                f"top{select_how_many}names"
            ].values
        )
        + list(
            pd.read_csv(f"./outputs/selected_images_max_{select_how_many}.csv")[
                f"top{select_how_many}names"
            ].values
        )
        + list(
            pd.read_csv(f"./outputs/selected_images_kbest_{select_how_many}.csv")[
                f"top{select_how_many}names"
            ].values
        )
    )
)
# %%
# remove highly correlated features
high_corr = list(
    pd.read_csv(f"./outputs/collinear_features_{select_how_many}.csv")[
        f"highcorrelation"
    ].values
)

select_images = remove_list_from_list(select_images, high_corr)

# %%

# Reduce image size and create 10m resolution images

os.makedirs("./outputs/selected_images_10m", exist_ok=True)

# delete old selected images
folder_path = "./outputs/selected_images_10m"

# Get a list of all files in the folder
file_list = os.listdir(folder_path)

# Loop through the file list and delete each file
for file_name in file_list:
    file_path = os.path.join(folder_path, file_name)
    if os.path.isfile(file_path):
        os.remove(file_path)


# get an EVI example
target_string = next((string for string in select_images if "EVI" in string), None)

for select_image in select_images:
    with gw.config.update(ref_image=target_string):
        with gw.open(
            select_image,
            nodata=9999,
            resampling="bilinear",
        ) as src:
            # replace missing with mean
            data = src.gw.replace({9999: src.mean()})
            data = data.gw.replace({np.nan: src.mean()})
            data.gw.save(
                f"./outputs/selected_images_10m/{os.path.basename(select_image)}",
                overwrite=True,
                compress="lzw",
            )


# %%
########################################################
# MODEL SELECTION between RF, LGBM, SVC
########################################################
# uses select_how_many from top of script

select_images = glob("./outputs/selected_images_10m/*.tif")

with gw.open(select_images, nodata=9999, stack_dim="band") as src:
    # fit a model to get Xy used to train model
    df = gw.extract(src, lu_poly, verbose=1)
    y = df["lc"]
    X = df[range(1, len(select_images) + 1)]
    X.columns = [os.path.basename(f).split(".")[0] for f in select_images]
    groups = df.id.values

X = pipeline_scale.fit_transform(X)
# %%

#  Create optuna classifier study

# Create an SQLite connection
conn = sqlite3.connect("models/study.db")

# Create a study with SQLite storage
storage = optuna.storages.RDBStorage(url="sqlite:///models/study.db")

# delete any existing study
try:
    study = optuna.load_study(study_name="model_selection", storage=storage)
    optuna.delete_study(study_name="model_selection", storage=storage)
except:
    pass

# create new study
study = optuna.create_study(
    storage=storage, study_name="model_selection", direction="maximize"
)


# Optimize the objective function
study.optimize(
    lambda trial: classifier_objective(
        trial,
        X,
        y,
        groups=groups,
        classifier_override=["LGBM", "RandomForest"],
        weights=df.Field_size,
    ),
    n_trials=150,
    n_jobs=12,
)

# Close the SQLite connection
conn.close()

print("Number of finished trials: {}".format(len(study.trials)))

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

# write params to csv
pd.DataFrame(study.trials_dataframe()).to_csv("models/optuna_study_model_selection.csv")


# Save results
conn = sqlite3.connect("models/study.db")

study = optuna.load_study(
    storage="sqlite:///models/study.db",
    study_name="model_selection",
)


# Access the top trials
top_trials = study.best_trials

# Iterate over the top trials and print their scores and parameters
for i, trial in enumerate(top_trials):
    print(f"Rank {i+1}: Score = {trial.value}")
    print(f"Parameters: {trial.params}")

# Get the DataFrame of all trials
trials_df = study.trials_dataframe()

# Sort the trials by the objective value in ascending order
sorted_trials = trials_df.sort_values("value", ascending=False)

# Print the ranked listing of trials
print(sorted_trials[["number", "value", "params_classifier"]])


##################################################################
# %%
########################################################
# Final Model & Class level prediction performance
########################################################
# NOTE: Performs better without kmeans included

# get optimal parameters
pipeline_performance = best_classifier_pipe("models/study.db", "model_selection")
print(pipeline_performance)

# get important image paths
# select_images = get_selected_ranked_images(
#     original_rank_images_df=f"./outputs/selected_images_{select_how_many}.csv",
#     subset_image_list=glob("./outputs/selected_images_10m/*.tif"),
#     select_how_many=select_how_many,
# )

select_images = glob("./outputs/selected_images_10m/*.tif")

select_images = select_images

# Get the image names
image_names = [os.path.basename(f).split(".")[0] for f in select_images]
select_images

# %%
# extract data
# with gw.open(select_images, nodata=9999, stack_dim="band") as src:
#     # fit a model to get Xy used to train model
#     X = gw.extract(src, lu_complete)
#     y = lu_complete["lc"]
#     y.reset_index(drop=True, inplace=True)
#     X = X[range(1, len(select_images) + 1)]
#     X.columns = [os.path.basename(f).split(".")[0] for f in select_images]

with gw.open(select_images, nodata=9999, stack_dim="band") as src:
    # fit a model to get Xy used to train model
    df = gw.extract(src, lu_poly, verbose=1)
    y = df["lc"]
    X = df[range(1, len(select_images) + 1)]
    X.columns = [os.path.basename(f).split(".")[0] for f in select_images]
    groups = df.id.values
    weights = df.Field_size


X = pipeline_scale.fit_transform(X)

# %%
# generate confusion matrix out of sample
conf_matrix_list_of_arrays = []
list_balanced_accuracy = []
list_kappa = []
# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
for i, (train_index, test_index) in enumerate(skf.split(X, y, groups)):
    # for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    pipeline_performance.fit(
        X_train, y_train, classifier__sample_weight=weights[train_index]
    )
    y_pred = pipeline_performance.predict(X_test)

    # get performance metrics
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    list_balanced_accuracy.append(balanced_accuracy)

    kappa_accuracy = cohen_kappa_score(y_test, y_pred)
    list_kappa.append(kappa_accuracy)

    # Get the class names from the label encoder
    class_names = pipeline_performance[
        "classifier"
    ].classes_  # le.inverse_transform(pipeline_performance["classifier"].classes_)

    # Create the confusion matrix with class names as row and column index
    conf_matrix = confusion_matrix(y_test, y_pred, labels=class_names)
    conf_matrix_list_of_arrays.append(conf_matrix)

conf_matrix_list_of_arrays

# get aggregate confusion matrix
agg_conf_matrix = np.sum(conf_matrix_list_of_arrays, axis=0)
balanced_accuracy = np.array(list_balanced_accuracy).mean()
kappa_accuracy = np.array(list_kappa).mean()

# Calculate the row-wise sums
row_sums = agg_conf_matrix.sum(axis=1, keepdims=True)

# Convert counts to percentages by row
conf_matrix_percent = agg_conf_matrix / row_sums

# Get the class names
class_names = le.inverse_transform(pipeline_performance["classifier"].classes_)
# Create a heatmap using seaborn
plt.figure(figsize=(10, 8))

sns.heatmap(
    conf_matrix_percent,
    annot=True,
    cmap="Blues",
    fmt=".0%",
    xticklabels=class_names,
    yticklabels=class_names,
)

# Set labels and title
plt.xlabel("Predicted")
plt.ylabel("True")
# plt.title(f"RF Confusion Matrix: Balance Accuracy = {round(balanced_accuracy, 2)}")
plt.title(f"Out of Sample Mean Confusion Matrix: Kappa = {round(kappa_accuracy, 2)}")
plt.savefig(
    f"outputs/final_class_perfomance_rf_kbest_{select_how_many}.png",
    bbox_inches="tight",
)

# Show the plot
plt.show()

# # %%
# # generate in sample confusion matrix

# pipeline_performance.fit(X, y, classifier__sample_weight=weights)
# y_pred = pipeline_performance.predict(X)
# # %%
# # get performance metrics
# balanced_accuracy = balanced_accuracy_score(y, y_pred)
# kappa_accuracy = cohen_kappa_score(y, y_pred)

# # Get the class names
# class_names = pipeline_performance[
#     "classifier"
# ].classes_  # le.inverse_transform(pipeline_performance["classifier"].classes_)

# # Create the confusion matrix with class names as row and column index
# conf_matrix = confusion_matrix(y, y_pred, labels=class_names)

# # Calculate the row-wise sums
# row_sums = agg_conf_matrix.sum(axis=1, keepdims=True)

# # Convert counts to percentages by row
# conf_matrix_percent = conf_matrix / row_sums

# # Create a heatmap using seaborn
# plt.figure(figsize=(10, 8))

# sns.heatmap(
#     conf_matrix_percent,
#     annot=True,
#     cmap="Blues",
#     fmt=".0%",
#     xticklabels=class_names,
#     yticklabels=class_names,
# )

# # Set labels and title
# plt.xlabel("Predicted")
# plt.ylabel("True")
# # plt.title(f"RF Confusion Matrix: Balance Accuracy = {round(balanced_accuracy, 2)}")
# plt.title(f"In Sample Confusion Matrix: Kappa = {round(kappa_accuracy, 2)}")
# plt.savefig(
#     f"outputs/final_class_perfomance_rf_kbest_{select_how_many}.png",
#     bbox_inches="tight",
# )

# # Show the plot
# plt.show()

# %%
##################################################################
# Write out final mode
##################################################################
# # %%

# %% Create a prediction stack

with gw.open(select_images, nodata=9999, stack_dim="band") as src:
    src = pipeline_scale_clean.fit_transform(src)

    src.gw.save(
        "outputs/pred_stack.tif",
        compress="lzw",
        overwrite=True,  # bigtiff=True
    )


# %%
# predict to stack
def user_func(w, block, model):
    pred_shape = list(block.shape)
    X = block.reshape(pred_shape[0], -1).T
    pred_shape[0] = 1
    y_hat = model.predict(X)
    X_reshaped = y_hat.T.reshape(pred_shape)
    return w, X_reshaped


gw.apply(
    "outputs/pred_stack.tif",
    f"outputs/final_model_rf{len(select_images)}.tif",
    user_func,
    args=(pipeline_performance,),
    n_jobs=16,
    count=1,
)


###############################################
# %% Compare to unsupervised

files = glob("./outputs/*kmean*.tif")
with gw.open(
    files,
    nodata=9999,
    stack_dim="band",
) as src:
    # fit a model to get Xy used to train model
    X = gw.extract(src, lu_complete)

    y = lu_complete["lc"]

for i in range(0, len(files)):
    print(files[i])

    y_hat = X[i + 1]
    y_hat = np.reshape(y_hat, (-1, 1))  # Reshape to (742, 1)
    # Create an instance of RandomForestClassifier
    rf_classifier = RandomForestClassifier()

    # Fit the classifier to your training data
    rf_classifier.fit(y_hat, y)

    # Predict the labels for your training data
    y_pred = rf_classifier.predict(y_hat)

    # Calculate the balanced accuracy score for the training data
    print(files[i])
    print(f"Kapa accuracy: {cohen_kappa_score(y, y_pred)}")

    conf_matrix = confusion_matrix(
        y,
        y_pred,  # labels=le.inverse_transform(rf_classifier.classes_)
    )

    # Calculate the row-wise sums
    row_sums = conf_matrix.sum(axis=1, keepdims=True)

    # Convert counts to percentages by row
    conf_matrix_percent = conf_matrix / row_sums

    # Get the class names
    class_names = le.inverse_transform(rf_classifier.classes_)

    # Create a heatmap using seaborn
    plt.figure(figsize=(10, 8))

    sns.heatmap(
        conf_matrix_percent,
        annot=True,
        cmap="Blues",
        fmt=".0%",
        xticklabels=class_names,
        yticklabels=class_names,
    )

    # Set labels and title
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(
        f"Confusion Matrix Kmean: {files[i]} \n Kappa Accuracy = {round(cohen_kappa_score(y, y_pred),3)}"
    )
    plt.savefig(
        f"outputs/final_class_perfomance_{os.path.basename(files[i])}.png",
        bbox_inches="tight",
    )

    # Show the plot
    plt.show()

# %%

# num_splits = 5
# # Initialize a dictionary to store the accuracies for each class
# class_accuracies = {}

# # Perform the train-test splits and compute accuracies for each class
# for i in range(num_splits):
#     print(f"Split {i+1}/{num_splits}")
#     # Split the data into train and test sets, stratified by the class
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, stratify=y, random_state=i
#     )

#     # Fit the classifier on the training data
#     pipeline.fit(X_train, y_train)

#     # Predict the labels for the test data
#     y_pred = pipeline.predict(X_test)

#     # Compute the accuracy for each class
#     accuracies = balanced_accuracy_score(y_test, y_pred)

#     # Store the accuracies in the dictionary
#     class_id = pipeline["classifier"].classes_
#     for class_label, class_name, accuracy in zip(
#         class_id, le.inverse_transform(class_id), accuracies
#     ):
#         if class_label not in class_accuracies:
#             class_accuracies[class_name] = []
#         class_accuracies[class_name].append(accuracy)

#     # # Store the accuracies in the dictionary
#     # for class_label, accuracy in zip(pipeline["classifier"].classes_, accuracies):
#     #     if class_label not in class_accuracies:
#     #         class_accuracies[class_label] = []
#     #     class_accuracies[class_label].append(accuracy)
# %%
# Print the accuracies for each class
for class_label, accuracies in class_accuracies.items():
    print(f"Class: {class_label}, Accuracies: {accuracies}")


# %%
# 55
# %%

# Get the confusion matrix
cm = confusion_matrix(y_test, pipeline.predict(X_test))

# We will store the results in a dictionary for easy access later
per_class_accuracies = {}

# Calculate the accuracy for each one of our classes
for idx, cls in enumerate(np.unique(y_test)):
    # True negatives are all the samples that are not our current GT class (not the current row)
    # and were not predicted as the current class (not the current column)
    true_negatives = np.sum(np.delete(np.delete(cm, idx, axis=0), idx, axis=1))

    # True positives are all the samples of our current GT class that were predicted as such
    true_positives = cm[idx, idx]

    # The accuracy for the current class is the ratio between correct predictions to all predictions
    per_class_accuracies[cls] = (true_positives + true_negatives) / np.sum(cm)
per_class_accuracies


# %%
###########################################################
# MODEL PREDICTION   Use environment crop_pred
###########################################################


components = 7
neighbors = 5

# get important image paths
select_images = get_selected_ranked_images(
    original_rank_images_df=f"./outputs/selected_images_{select_how_many}.csv",
    subset_image_list=glob("./outputs/selected_images_10m/*.tif"),
    select_how_many=select_how_many,
)
# add unsupervised classification images
select_images = select_images[
    0:10
]  # + glob("./outputs/*kmean*.tif") # kmeans might not help
print(select_images)

# Get the image names
image_names = [os.path.basename(f).split(".")[0] for f in select_images]


with gw.open(select_images, nodata=9999, stack_dim="band") as src:
    # fit a model to get Xy used to train model
    X = gw.extract(src, lu_complete)
    y = lu_complete["lc"]
    X = X[range(1, len(select_images) + 1)]
    X.columns = [os.path.basename(f).split(".")[0] for f in select_images]
# %%
# Define the pipeline steps
pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        (
            "umap",
            umap.UMAP(
                n_components=components,
                low_memory=True,
                random_state=42,
                n_neighbors=neighbors,
                n_jobs=-1,
            ),
        ),
        # ("pca", PCA(n_components=5)),
        ("classifier", RandomForestClassifier(n_estimators=500)),
    ]
)


# Define the parameter grid for RandomizedSearchCV
param_grid = {
    "classifier__n_estimators": sp_randint(100, 1000),
    "classifier__max_depth": sp_randint(5, 20),
    "classifier__min_samples_split": sp_randint(2, 10),
    "classifier__min_samples_leaf": sp_randint(1, 5),
    "classifier__max_features": ["sqrt", "log2"],
    "classifier__bootstrap": [True, False],
}

# Create the RandomizedSearchCV object with stratified cross-validation
search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_grid,
    n_iter=10,
    cv=StratifiedKFold(n_splits=5),
    scoring="balanced_accuracy",
)

# Fit the data to perform the search
search.fit(X, y)

# Access the best parameters and best score
best_params = search.best_params_
best_score = search.best_score_
# %%
# Save the trained model

with open(f"models/final_model_rf_{len(select_images)}.pkl", "wb") as file:
    pickle.dump(search, file)
# save best params
pd.DataFrame(best_params, index=pd.Index([0])).to_csv(
    f"models/best_params_rf_{len(select_images)}.csv"
)
# save class names

pd.DataFrame(
    {"class": search.classes_, "Names": le.inverse_transform(search.classes_)}
).to_csv(f"models/class_names_rf_{len(select_images)}.csv")


# %% Load the saved model
import pickle

with open(f"models/final_model_rf_{len(select_images)}.pkl", "rb") as file:
    search = pickle.load(file)


# %% Create a prediction stack

with gw.open(select_images, nodata=9999, stack_dim="band") as src:
    src.gw.save(
        "outputs/pred_stack.tif",
        compress="lzw",
        overwrite=True,  # bigtiff=True
    )


# %% Predict to the stack


def user_func(w, block, model):
    pred_shape = list(block.shape)
    X = block.reshape(pred_shape[0], -1).T
    pred_shape[0] = 1
    y_hat = model.predict(X)
    X_reshaped = y_hat.T.reshape(pred_shape)
    return w, X_reshaped


gw.apply(
    "outputs/pred_stack.tif",
    f"outputs/final_model_rf{len(select_images)}.tif",
    user_func,
    args=(search.best_estimator_,),
    n_jobs=16,
    count=1,
)


# %
############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################

# %%
# GroupKFold
cv = CrossValidatorWrapper(KFold(n_splits=3))
gridsearch2 = GridSearchCV(
    pl,
    cv=cv,
    scoring="balanced_accuracy",
    param_grid={"clf__n_estimators": [500]},
)

# get an EVI example
target_string = next((string for string in select_images if "EVI" in string), None)

import cProfile, pstats

profiler = cProfile.Profile()
profiler.enable()

with gw.config.update(ref_image=target_string):
    with gw.open(select_images, nodata=9999, stack_dim="band") as src:
        # src = src.gw.mask_nodata()
        # fit a model to get Xy used to train model

        X, Xy, outpipe = fit(data=src, clf=pl, labels=lu_complete, col="lc")
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats("ncalls")
        stats.print_stats()

        # %%
        # fit cross valiation and parameter tuning
        gridsearch2.fit(*Xy)
        print(gridsearch2.cv_results_)
        print(gridsearch2.best_score_)

        outpipe.set_params(**gridsearch2.best_params_)
        # print("predcting:")
        y = predict(src, X, outpipe)
        # print(y.values)
        # print(np.nanmax(y.values))
        # y.plot(robust=True, ax=ax)
y.gw.save(
    "/home/mmann1123/extra_space/Dropbox/Tanzania_data/Projects/YM_Tanzania_Field_Boundaries/Land_Cover/outputs/ym_prediction.tif",
    nodata=9999,
)
# plt.tight_layout(pad=1)
# print("plotting")
# for i in range(src.shape[0]):
#     fig, ax = plt.subplots(dpi=200, figsize=(5, 5))
#     src[i].plot(robust=True, ax=ax)
#     plt.tight_layout(pad=1)
# %% Assess performance

# %%
from sklearn.metrics import confusion_matrix

# Assuming you have a fitted GridSearchCV object named 'grid_search'
best_estimator = gridsearch2.best_estimator_
X1, y1 = *Xy
# Make predictions on the test data
y_pred = best_estimator.predict(Xy[0])

# Calculate the confusion matrix
mat = confusion_matrix(y_true, y_pred)
sns.heatmap(mat.T, square=True, annot=True, fmt="d", cbar=False)
plt.xlabel("true label")
plt.ylabel("predicted label")

# %%

from sklearn.model_selection import cross_val_predict

y_pred = cross_val_predict(pl, *Xy, cv=3)


# %%
# # create kfold by
# # groupkfold max score 0.003
# # gkf = list(GroupKFold(n_splits=5).split(X_sorghum, y_sorghum, groups=X_sorghum.index))
# # groupshufflesplit max score 0.8 ish
# gkf = list(
#     GroupShuffleSplit(n_splits=5).split(X_sorghum, y_sorghum, groups=X_sorghum.index)
# )
# # by year no improvement
# # gkf = list(
# #     GroupKFold(n_splits=3).split(X_sorghum, y_sorghum, groups=X_sorghum.year)
# # )


# # break into treatment by numeric and categorical
# numeric_features = list(
#     X_sorghum.select_dtypes(include=["int64", "float32", "float64"]).columns
# )
# categorical_features = list(X_sorghum.select_dtypes(include=["object"]).columns)


# # set up pipelines for preprocessing categorical and numeric data
# numeric_transformer = Pipeline(
#     steps=[
#         (
#             "imputer",
#             SimpleImputer(strategy="median"),
#         ),  # scale not needed for trees("scaler", StandardScaler())
#     ]
# )

# categorical_transformer = Pipeline(
#     steps=[
#         ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
#         ("onehot", OneHotEncoder(handle_unknown="ignore")),
#     ]
# )

# # define preprocessor
# preprocess = ColumnTransformer(
#     transformers=[
#         ("num", numeric_transformer, numeric_features),
#         ("cat", categorical_transformer, categorical_features),
#     ]
# )

# full_pipe = Pipeline(
#     steps=[
#         ("preprocess", preprocess),
#         # ("pca", MiniBatchSparsePCA()),
#         ("lgbm", LGBMRegressor(random_state=42)),
#     ]
# )


# depth = [int(x) for x in np.linspace(5, 50, num=11)]
# depth.append(None)

# random_grid = {
#     # light gradient boosting
#     "lgbm__objective": ["regression"],
#     # 'poisson' 'regression', huber&fair is less impaced by outliers than MSE https://heartbeat.fritz.ai/5-regression-loss-functions-all-machine-learners-should-know-4fb140e9d4b0
#     "lgbm__n_estimators": [int(x) for x in np.linspace(100, 3000, num=10)],  #
#     "lgbm__max_depth": [-1, 2, 10, 100],
#     "lgbm__min_data_in_leaf": [int(x) for x in np.linspace(1, 500, num=5)],
#     # "lgbm__num_leaves": [int(x) for x in np.linspace(1, 2 ^ (100), num=10)],
#     # keep less than 2^(max_depth)
#     "lgbm__device_type": ["cpu"],
#     "lgbm__bagging_fraction": [0.75, 1],  #
#     "lgbm__poisson_max_delta_step": [
#         int(x) for x in np.linspace(0.1, 3.0, num=10)
#     ],  # might be same as lambda??
# }


# grid_search = RandomizedSearchCV(
#     full_pipe,
#     random_grid,
#     cv=gkf,
#     n_jobs=4,
#     verbose=1000,
#     return_train_score=False,
#     # "pca__n_components": [10],
#     scoring="r2",
#     n_iter=3,
#     random_state=1,
# )  # 10number of random draws x # folds for total jobs

# model = grid_search.fit(X_sorghum, y_sorghum)

# # %%
# print("R2:", r2_score(y_sorghum, model.predict(X_sorghum)))
# print("best score", model.best_score_)


# d = {
#     "variable": X_sorghum.columns,
#     "importance": model.best_estimator_.named_steps["lgbm"].feature_importances_,
# }

# df = pd.DataFrame(data=d)
# df.sort_values(by=["importance"], ascending=False, inplace=True)
# print(df)

# # %%
# df.to_csv(
#     os.path.join(
#         data_path,
#         "Projects/ET_ATA_Crops/models/rf_variable_importance_yield_Xy_11_15_18_mike.csv",
#     ),
# )

# # %%
# results_in_splits = []

# for k, v in model.cv_results_.items():
#     if "split" in k:
#         print("\t->", k)
#         results_in_splits.append(v[0])
#     else:
#         print(k)

# print("\n")
# print(sum(results_in_splits) / len(results_in_splits))
# print(model.best_score_)

# # %%

# %%
