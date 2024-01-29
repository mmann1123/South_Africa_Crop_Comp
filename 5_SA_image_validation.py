# %% create grids of images of training data

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import urllib
import PIL


# NOTES: Millet, Sorghum, maize very similar especially when young,
# maybe combine millet and sorghum into one category
# young plants typically labeled as maize


os.chdir(
    "/home/mmann1123/extra_space/Dropbox/Tanzania_data/Projects/YM_Tanzania_Field_Boundaries/Land_Cover"
)
# Read the CSV file
df = pd.read_csv("../kobo_field_collections/TZ_ground_truth_cleaned_ls.csv")
df = df.reset_index()
df_filtered = df.dropna(subset=["Picture_of_the_field_or_feature_URL", "primar"])
df_filtered = df_filtered[
    [
        "index",
        "Picture_of_the_field_or_feature_URL",
        "primar",
        "Secondary_land_cover",
        "Quality",
    ]
]
df_filtered

# %% Subset of images
# Filter rows with valid URLs and assigned crop labels
for crop in df_filtered["primar"].unique():
    try:
        df_sample = df_filtered[df_filtered["primar"] == crop].sample(
            n=12, random_state=1
        )
    except:
        df_sample = df_filtered[df_filtered["primar"] == crop]

    df_sample = df_sample.reset_index(drop=True)
    # Define the grid size
    # Calculate the number of rows based on the number of images
    num_cols = 3
    num_images = len(df_sample)
    try:
        num_rows = (num_images - 1) // num_cols + 1
    except:
        num_rows = 1

    # Create the grid of subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 16))

    # Iterate over the filtered rows and plot the images
    for idx, row in df_sample.iterrows():
        if idx >= num_rows * num_cols:
            break

        # Get the image URL and crop label
        img_url = row["Picture_of_the_field_or_feature_URL"]
        crop_label = row["primar"]

        # Load the image from URL
        # img = mpimg.imread(img_url)
        img = PIL.Image.open(urllib.request.urlopen(img_url))

        # Get the corresponding subplot
        if num_rows == 1:
            ax = axes[idx % num_cols]
        else:
            ax = axes[idx // num_cols, idx % num_cols]

        # Plot the image
        ax.imshow(img)
        ax.axis("off")
        # Add the crop label as a text
        ax.text(
            0.5,
            -0.1,
            crop_label,
            transform=ax.transAxes,
            horizontalalignment="center",
            verticalalignment="top",
            fontsize=12,
            color="white",
            backgroundcolor="black",
        )

    # Adjust the spacing between subplots
    plt.tight_layout()
    plt.savefig(f"outputs/crop_type_examples_{crop}.png", dpi=300)

    # Show the plot
    plt.show()


# %%
# %% Plot 100  images
# Filter rows with valid URLs and assigned crop labels

for crop in df_filtered["primar"].unique():
    try:
        df_sample = df_filtered[df_filtered["primar"] == crop].sample(
            n=102, random_state=1
        )
    except:
        df_sample = df_filtered[df_filtered["primar"] == crop]

    df_sample = df_sample.reset_index(drop=True)

    # Define the grid size
    # Calculate the number of rows based on the number of images
    num_cols = 3
    num_images = len(df_sample)
    try:
        num_rows = (num_images - 1) // num_cols + 1
    except:
        num_rows = 1

    # Create the grid of subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 3 * num_rows))

    # Iterate over the filtered rows and plot the images
    for idx, row in df_sample.iterrows():
        if idx >= num_rows * num_cols:
            break

        # Get the image URL and crop label
        img_url = row["Picture_of_the_field_or_feature_URL"]
        crop_label = row["primar"]

        # Load the image from URL
        # img = mpimg.imread(img_url)
        img = PIL.Image.open(urllib.request.urlopen(img_url))

        # Get the corresponding subplot
        if num_rows == 1:
            ax = axes[idx % num_cols]
        else:
            ax = axes[idx // num_cols, idx % num_cols]

        # Plot the image
        ax.imshow(img)
        ax.axis("off")
        # Add the crop label as a text
        ax.text(
            0.5,
            -0.1,
            f"{crop_label}, row {row['index']}",
            transform=ax.transAxes,
            horizontalalignment="center",
            verticalalignment="top",
            fontsize=12,
            color="white",
            backgroundcolor="black",
        )

    # Adjust the spacing between subplots
    plt.tight_layout()
    plt.savefig(f"outputs/crop_type_100_{crop}.png", dpi=300)

    # Show the plot
    plt.show()

# %% plot ALL images

keep = [
    "rice",
    "maize",
    "cassava",
    "sunflower",
    "sorghum",
    "cotton",
    "soybeans",
    "millet",
]
drop = [
    "Don't know",
    "Other (later, specify in optional notes)",
    "vegetables",
    "other",
    "speciality_crops",
    "eggplant",
    "okra ",
    "tree_crops",
    "other_grain",
]

# apply keep/drop
df_filtered.drop(df_filtered[df_filtered["primar"].isin(drop)].index, inplace=True)


for crop in df_filtered["primar"].unique():
    print("working on", crop)
    df_sample = df_filtered[df_filtered["primar"] == crop]

    df_sample = df_sample.reset_index(drop=True)

    # Define the grid size
    # Calculate the number of rows based on the number of images
    num_cols = 3
    num_images = len(df_sample)
    try:
        num_rows = (num_images - 1) // num_cols + 1
    except:
        num_rows = 1

    # Create the grid of subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 3 * num_rows))

    # Iterate over the filtered rows and plot the images
    for idx, row in df_sample.iterrows():
        if idx >= num_rows * num_cols:
            break

        # Get the image URL and crop label
        img_url = row["Picture_of_the_field_or_feature_URL"]
        crop_label = row["primar"]

        # Load the image from URL
        # img = mpimg.imread(img_url)
        try:
            img = PIL.Image.open(urllib.request.urlopen(img_url))
        except:
            continue
        # Get the corresponding subplot
        if num_rows == 1:
            ax = axes[idx % num_cols]
        else:
            ax = axes[idx // num_cols, idx % num_cols]

        # Plot the image
        ax.imshow(img)
        ax.axis("off")
        # Add the crop label as a text
        ax.text(
            0.5,
            -0.1,
            f"{crop_label}, row {row['index']}",
            transform=ax.transAxes,
            horizontalalignment="center",
            verticalalignment="top",
            fontsize=12,
            color="white",
            backgroundcolor="black",
        )

    # Adjust the spacing between subplots
    plt.tight_layout()


# %%
#############################################
# ERROR ANALYSIS
#############################################

os.chdir(
    "/home/mmann1123/extra_space/Dropbox/Tanzania_data/Projects/YM_Tanzania_Field_Boundaries/Land_Cover"
)
# Read the CSV file
df = pd.read_csv("../kobo_field_collections/TZ_ground_truth_cleaned_mm.csv")
df.columns
df.Quality.replace({"D": "Delete", "L": "Low", "M": "Medium"}, inplace=True)


df.groupby(["username"])["Quality"].count()
# %%
df.groupby(["username", "Quality"])["name"].count()
df.groupby(["username", "Quality"])["Quality"].count().to_csv(
    "./outputs/quality_counts.csv"
)

# %%
a = (
    df.groupby(["team_letter", "Quality"])["Quality"].count()
    / df.groupby(["team_letter"])["team_letter"].count()
)
a.to_csv("./outputs/quality_counts_teams.csv")
a
# %%

a = (
    df.groupby(["primar", "Quality"])["Quality"].count()
    / df.groupby(["primar"])["team_letter"].count()
)
a.to_csv("./outputs/quality_counts_crop.csv")
a

# %%

a = (
    df.groupby(["username", "Quality"])["Quality"].count()
    / df.groupby(["username"])["team_letter"].count()
)
a.to_csv("./outputs/quality_counts_username.csv")
a


##################################################
##################################################
# %% Examples of low quality images
##################################################
# Filter rows with valid URLs and assigned crop labels
for crop in ["D", "L", "M"]:
    try:
        df_sample = df_filtered[df_filtered["Quality"] == crop].sample(
            n=12, random_state=1
        )
    except:
        df_sample = df_filtered[df_filtered["Quality"] == crop]

    df_sample = df_sample.reset_index(drop=True)
    # Define the grid size
    # Calculate the number of rows based on the number of images
    num_cols = 3
    num_images = len(df_sample)
    try:
        num_rows = (num_images - 1) // num_cols + 1
    except:
        num_rows = 1

    # Create the grid of subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 16))
    crop_title = crop.replace("D", "Delete").replace("L", "Low").replace("M", "Medium")
    # fig.title(f"Image quality: {crop_title}")
    fig.suptitle(f"Image quality: {crop_title}", fontsize=20)

    # Iterate over the filtered rows and plot the images
    for idx, row in df_sample.iterrows():
        if idx >= num_rows * num_cols:
            break

        # Get the image URL and crop label
        img_url = row["Picture_of_the_field_or_feature_URL"]
        crop_label = row["primar"]

        # Load the image from URL
        # img = mpimg.imread(img_url)
        img = PIL.Image.open(urllib.request.urlopen(img_url))

        # Get the corresponding subplot
        if num_rows == 1:
            ax = axes[idx % num_cols]
        else:
            ax = axes[idx // num_cols, idx % num_cols]

        # Plot the image
        ax.imshow(img)
        ax.axis("off")
        # Add the crop label as a text
        ax.text(
            0.5,
            -0.1,
            crop_label,
            transform=ax.transAxes,
            horizontalalignment="center",
            verticalalignment="top",
            fontsize=12,
            color="white",
            backgroundcolor="black",
        )

    # Adjust the spacing between subplots
    plt.tight_layout()
    plt.savefig(f"outputs/quality_type_examples_{crop_title}.png", dpi=300)

    # Show the plot
    plt.show()

# %%
# create seaborn histogram of secondary_land_cover with vertical names on x-axis


import seaborn as sns

df_filtered["Secondary_land_cover"].replace(
    {
        "vegetables_and_pulses__examples__eggplan": "vegetables_and_pulses",
        "maize__mahindi": "maize",
        "sunflower__alizeti": "sunflower",
        "other__later__specify_in_optional_notes": "other",
        "tree_crops__examples__banana__coconut__g": "tree_crops",
        "other_grains__examples__wheat__barley__o": "other_grains",
        "sorghum__mtama": "sorghum",
        "cotton__pamba": "cotton",
        "peanuts_ground_nuts__karanga": "ground_nuts",
        "rice__mpunga": "rice",
        "millet__ulezi": "millet",
        "specialty_crops__cocoa__coffee__tea__sug": "specialty_crops",
        "grassland_savanna": "grassland_savanna",
        "don_t_know": "don't_know",
    },
    inplace=True,
)

import seaborn as sns

secondary_count = df_filtered.groupby(["Secondary_land_cover"])[
    "Secondary_land_cover"
].count()
secondary_count = secondary_count.reset_index(name="count")

# Calculate the total number of observations
total_observations = secondary_count["count"].sum()

# Calculate the percentage for each category
secondary_count["percentage"] = (secondary_count["count"] / total_observations) * 100

# Sort the DataFrame by percentage in descending order
secondary_count.sort_values(by="percentage", ascending=False, inplace=True)


secondary_count.sort_values(by="percentage", inplace=True, ascending=False)
# Create the histogram plot
ax = sns.barplot(data=secondary_count, x="Secondary_land_cover", y="percentage")

# Rotate the x-axis labels vertically
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_xlabel("Secondary Land Cover")
plt.ylabel("Percentage of Total Observations")

plt.savefig(f"outputs/secondary_land_cover.png", dpi=300)

# Display the plot
plt.show()


# (
#     df_filtered.groupby(["Secondary_land_cover"])["Secondary_land_cover"].count()
#     / df["Secondary_land_cover"].count()
# )


# %%

# %%
# create subset file for validation


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import urllib
import PIL
import geopandas as gpd


os.chdir(
    "/home/mmann1123/extra_space/Dropbox/Tanzania_data/Projects/YM_Tanzania_Field_Boundaries/Land_Cover"
)
# Read the CSV file
df = pd.read_csv("../kobo_field_collections/TZ_ground_truth_cleaned_ls.csv")
df_filtered = df.dropna(subset=["Picture_of_the_field_or_feature_URL", "primar"])
df_filtered = gpd.GeoDataFrame(
    df_filtered,
    geometry=gpd.points_from_xy(
        x=df_filtered._field_center_longitude, y=df_filtered._field_center_latitude
    ),
    crs="EPSG:4326",
)

df_filtered = df_filtered[
    [
        "Picture_of_the_field_or_feature_URL",
        "primar",
        "Secondary_land_cover",
        "Field_size",
        "Quality_Drop_Low",
        "Lisa Notes",
        "geometry",
    ]
]


keep = [
    "rice",
    "maize",
    "cassava",
    "sunflower",
    "sorghum",
    "cotton",
    "soybeans",
    "millet",
]
drop = [
    "Don't know",
    "Other (later, specify in optional notes)",
    "vegetables",
    "other",
    "speciality_crops",
    "eggplant",
    "okra ",
    "tree_crops",
    "other_grain",
]

# apply keep/drop
df_filtered.drop(df_filtered[df_filtered["primar"].isin(drop)].index, inplace=True)
df_filtered.drop(
    df_filtered[df_filtered["Quality_Drop_Low"].isin(["D", "L"])].index, inplace=True
)
df_filtered["PleaseValidate"] = ""
df_filtered.reset_index(inplace=True, drop=True)
df_filtered.reset_index(inplace=True)

# %% break into 2 files


df_filtered.iloc[: len(df_filtered) // 2].to_file(
    f"../kobo_field_collections/TZ_ground_truth_cleaned_ls_p1.geojson",
    driver="GeoJSON",
)

df_filtered.iloc[len(df_filtered) // 2 :].to_file(
    f"../kobo_field_collections/TZ_ground_truth_cleaned_ls_p2.geojson",
    driver="GeoJSON",
)

# %%
