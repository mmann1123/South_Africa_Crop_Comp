# %%
# create seaborn histogram of secondary_land_cover with vertical names on x-axis


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
        "Quality_Drop_Low",
    ]
]
df_filtered
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
ax.set_xlabel("Secondary Land Cover", fontsize=15)

# increas font size for x and y axis and axis labels
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)


plt.ylabel("Percentage of Total Observations", fontsize=15)


plt.savefig(f"outputs/secondary_land_cover.png", dpi=300)

# Display the plot
plt.show()


# (
#     df_filtered.groupby(["Secondary_land_cover"])["Secondary_land_cover"].count()
#     / df["Secondary_land_cover"].count()
# )


# %% ghant chart of months Dont use... google charts is better

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Create a DataFrame with the start and end months of each growing season
data = {
    "Season": ["Masika", "Vuli", "Msimu"],
    "Start_Month": ["March", "September", "November"],
    "End_Month": ["August", "February", "June"],
}
seasons_df = pd.DataFrame(data)

# Convert month names to numerical values for sorting
month_order = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
seasons_df["Start_Month"] = pd.Categorical(
    seasons_df["Start_Month"], categories=month_order, ordered=True
)
seasons_df["End_Month"] = pd.Categorical(
    seasons_df["End_Month"], categories=month_order, ordered=True
)

# Sort the DataFrame by start month
seasons_df = seasons_df.sort_values(by="Start_Month")

# Define the colors for each growing season
season_colors = {"Masika": "blue", "Vuli": "green", "Msimu": "orange"}

# Create the Gantt chart using hlines with customizations
plt.figure(figsize=(10, 3))
for index, row in seasons_df.iterrows():
    plt.hlines(
        y=row["Season"],
        xmin=row["Start_Month"],
        xmax=row["End_Month"],
        color=season_colors[row["Season"]],
        lw=3,  # Thicker lines
    )

# Customize the plot
plt.xlabel("Month")
plt.ylabel("Growing Season")
plt.title("Monthly Timing of Growing Seasons in Tanzania")
plt.yticks(ticks=seasons_df.index, labels=seasons_df["Season"])
plt.xticks(
    ticks=range(len(month_order)), labels=[month[:3] for month in month_order]
)  # Abbreviate month names
plt.tight_layout()
plt.subplots_adjust(hspace=0.1)  # Reduce space between y-axis ticks
plt.show()
