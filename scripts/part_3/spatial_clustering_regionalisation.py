# ==============================================================================
#             CHAPTER 8 - SPATIAL CLUSTERING  AND REGIONALISATION
# ==============================================================================


# ------------------------------------------------------------------------------
#                      GET SAMPLE DATA FROM THE US CENSUS
# ------------------------------------------------------------------------------


## ---- Load required libraries ------------------------------------------------


from cmath import polar
from scripts.part_1.working_with_crs import fig
from census import Census
from us import states
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from libpysal.weights import Queen, KNN
from esda.moran import Moran
import numpy as np
from sklearn.preprocessing import robust_scale
from sklearn.cluster import KMeans, AgglomerativeClustering
import geoplot as gplt
import plotly.graph_objs as go

## ---- Pull data from the Annual Community Survey  ----------------------------

c = Census("482adb2a158e378bd3d169970cd113448c2dcdd7")
### List of variables ----
geo_demo = [
    "B01003_001E",  # "Total Population"
    "B25077_001E",  # "Median value of owner occupied units"
    "B25026_001E",  # "Total population in occupied housing units"
    "B25008_002E",  # "Total number of owner occupied units"
    "B25008_003E",  # "Total number of renter occupied units"
    "B06009_002E",  # "Population with less than a high school diploma"
    "B06009_003E",  # "Population with high school diploma or equivalent"
    "B06009_004E",  # "Population with some college/associates degree"
    "B06009_005E",  # "Population with bachelors degree"
    "B06009_006E",  # "Population with a graduate degree"
    "B01002_001E",  # "Median age"
    "B06010_004E",  # "Population with income less than 9999"
    "B06010_005E",  # "Population with income between 10000 and 14999"
    "B06010_006E",  # "Population with income between 15000 and 24999"
    "B06010_007E",  # "Population with income between 25000 and 34999"
    "B06010_008E",  # "Population with income between 35000 and 49999"
    "B06010_009E",  # "Population with income between 50000 and 64999"
    "B06010_010E",  # "Population with income between 65000 and 74999"
    "B06010_011E",  # "Population with income of 75000 or more"
    "B28007_009E",  # "Population in labor force and unemployed"
    "B19059_002E",  # "Population that is retired with retirement income"
    "B19059_003E",  # "Retired without retirement income"
    "B08013_001E",  # "Travel time to work in minutes"
    "B17013_002E",  # "Population with income below poverty level in past 12 months"
]

### Get data ----
ny_census = c.acs5.state_county_tract(
    fields=(
        "NAME",
        "B01003_001E",
        "B25026_001E",
        "B25008_002E",
        "B25008_003E",
        "B25077_001E",
        "B06009_002E",
        "B06009_003E",
        "B06009_004E",
        "B06009_005E",
        "B06009_006E",
        "B01002_001E",
        "B06010_004E",
        "B06010_005E",
        "B06010_006E",
        "B06010_007E",
        "B06010_008E",
        "B06010_009E",
        "B06010_010E",
        "B06010_011E",
        "B28007_009E",
        "B19059_002E",
        "B19059_003E",
        "B08013_001E",
        "B17013_002E",
    ),
    state_fips=states.NY.fips,
    county_fips="*",
    tract="*",
    year=2019,
)

### Create a dataframe from the census data ----
ny_df = pd.DataFrame(ny_census)

### Access shapefile of NY census tracts ----
ny_tract = gpd.read_file(
    "https://www2.census.gov/geo/tiger/TIGER2019/TRACT/tl_2019_36_tract.zip"
)

### Reprojecting the shapefile to the NY State Plan Long Island Zone ESPG:2263
ny_tract.to_crs("EPSG:2263", inplace=True)

### Create a GEOID variable ----
ny_df["GEOID"] = ny_df["state"] + ny_df["county"] + ny_df["tract"]

### Remove the individual columns as they're no longer needed ----
ny_df.drop(columns=["state", "county", "tract"], inplace=True)

# Join the data together on GEOID to geoenable the census data
ny_merge = ny_tract.merge(ny_df, on="GEOID")


# Renaming variables in the data set
ny_merge.rename(
    columns={
        "B01003_001E": "TotPop",  # "Total Population"
        "B25077_001E": "MedVal_OwnOccUnit",  # "Median value of owner occupied units"
        "B25026_001E": "TotPopOccUnits",  # "Total population in occupied housing units"
        "B25008_002E": "TotNumOwnOccUnit",  # "Total number of owner occupied units"
        "B25008_003E": "TotNumRentOccUnit",  # "Total number of renter occupied units"
        "B06009_002E": "PopLTHSDip",  # "Population with less than a high school diploma"
        "B06009_003E": "PopHSDip",  # "Population with high school diploma or equivalent"
        "B06009_004E": "PopAssoc",  # "Population with some college/associates degree"
        "B06009_005E": "PopBA",  # "Population with bachelors degree"
        "B06009_006E": "PopGrad",  # "Population with a graduate degree"
        "B01002_001E": "MedAge",  # "Median age"
        "B06010_004E": "PopIncLT10",  # "Population with income less than 9999"
        "B06010_005E": "PopInc1015",  # "Population with income between 10000 and 14999"
        "B06010_006E": "PopInc1525",  # "Population with income between 15000 and 24999"
        "B06010_007E": "PopInc2535",  # "Population with income between 25000 and 34999"
        "B06010_008E": "PopInc3550",  # "Population with income between 35000 and 49999"
        "B06010_009E": "PopInc5065",  # "Population with income between 50000 and 64999"
        "B06010_010E": "PopInc6575",  # "Population with income between 65000 and 74999"
        "B06010_011E": "PopIncGT75",  # "Population with income of 75000 or more"
        "B28007_009E": "UnempPop",  # "Population in labor force and unemployed"
        "B19059_002E": "RetPop",  # "Population that is retired with retirement income"
        "B19059_003E": "RetPopNoRetInc",  # "Retired without retirement income"
        "B08013_001E": "TrvTimWrk",  # "Travel time to work in minutes"
        "B17013_002E": "PopBlwPovLvl",  # "Population with income below poverty level in past 12 months"
    },
    inplace=True,
)


geo_demo_rn = [
    "TotPop",  # "Total Population"
    "TotPopOccUnits",  # "Total population in occupied housing units"
    "TotNumOwnOccUnit",  # "Total number of owner occupied units"
    "TotNumRentOccUnit",  # "Total number of renter occupied units"
    "PopLTHSDip",  # "Population with less than a high school diploma"
    "PopHSDip",  # "Population with high school diploma or equivalent"
    "PopAssoc",  # "Population with some college/associates degree"
    "PopBA",  # "Population with bachelors degree"
    "PopGrad",  # "Population with a graduate degree"
    "PopIncLT10",  # "Population with income less than 9999"
    "PopInc1015",  # "Population with income between 10000 and 14999"
    "PopInc1525",  # "Population with income between 15000 and 24999"
    "PopInc2535",  # "Population with income between 25000 and 34999"
    "PopInc3550",  # "Population with income between 35000 and 49999"
    "PopInc5065",  # "Population with income between 50000 and 64999"
    "PopInc6575",  # "Population with income between 65000 and 74999"
    "PopIncGT75",  # "Population with income of 75000 or more"
    "UnempPop",  # "Population in labor force and unemployed"
    "RetPop",  # "Population that is retired with retirement income"
    "RetPopNoRetInc",  # "Retired without retirement income"
    "PopBlwPovLvl",  # "Population with income below poverty level in past 12 months"
]

### Cleaning up the dataframe ----
geo_demo_rn.append("geometry")
ny_merge_2 = ny_merge[geo_demo_rn]
geo_demo_rn.remove("geometry")

### Dropping any areas without population ----
ny_merge_2 = ny_merge_2[ny_merge_2["TotPop"] > 0]

### Resetting the index to assist in index based operations later on ----
ny_merge_2.reset_index(inplace=True)


## ---- Conduct Exploratory Data Analysis --------------------------------------


### Plot a map of each extracted variables from the Census API ----
fix, axes = plt.subplots(ncols=3, nrows=7, figsize=(75, 75), layout="tight")
axes = axes.flatten()
plt.rcParams["font.size"] = "40"

#### Iterate over the list of variables ----
for ind, col in enumerate(geo_demo_rn):
    ax = axes[ind]
    ny_merge_2.plot(
        column=col,
        ax=ax,
        scheme="quantiles",
        linewidth=2,
        cmap="coolwarm",
        legend=True,
        legend_kwds={"loc": "center left", "bbox_to_anchor": (2, 0.5), "fmt": "{:.0f}"},
    )
    ax.set_axis_off()
    ax.set_title(col)
    plt.subplots_adjust(wspace=None, hspace=None)
    plt.show()


## ---- Measuring Spatial Autocorrelation --------------------------------------


### Calculate Queen spatial weights matrix ----
w = Queen.from_dataframe(ny_merge_2)
np.random.seed(54321)

### Calculate Moran's I index for each variable ----
moransi_results = [Moran(ny_merge_2[i], w) for i in geo_demo_rn]

moransi_results = [
    (v, res.I, res.p_sim) for v, res in zip(geo_demo_rn, moransi_results)
]

table = pd.DataFrame(
    moransi_results, columns=["GEODEMO Var", "Moran's I", "P-value"]
).set_index("GEODEMO Var")


## ---- Scale the data ---------------------------------------------------------


ny_merged_scaled = robust_scale(ny_merge_2[geo_demo_rn])


## ---- K-Means Clustering -----------------------------------------------------


np.random.seed(54321)

### Elbow plot ----
distortions = []
K = range(1, 15)
for k in K:
    kmeans = KMeans(n_clusters=k).fit(ny_merged_scaled)
    distortions.append(kmeans.inertia_)
plt.figure(figsize=(40, 25))
plt.plot(K, distortions, "bx-")
plt.xlabel("Number of clusters")
plt.ylabel("Distortion")
plt.title("Elbow method for optimal k")
plt.show()

### K-Means = 5 Clusters ----
kmeans_5 = KMeans(n_clusters=5).fit(ny_merged_scaled)

### Visualise ----
### Visualise the distribution of price across census tract ----
ny_merge_2["kmeans_5_label"] = kmeans_5.labels_
f, ax = plt.subplots(1, figsize=(40, 20))
ny_merge_2.plot(
    ax=ax, column="kmeans_5_label", legend=True, categorical=True, linewidth=0.5
)

### Cluster profiling ----
#### Descriptive statistics of each cluster ----
kgdistr = ny_merge_2.groupby("kmeans_5_label").size()
k5means = ny_merge_2.groupby("kmeans_5_label")[geo_demo_rn].mean().round(2)

### Plot a cluster radial plot ----
#### Create a dataframe of scaled data ----
ny_merged_scaled_df = pd.DataFrame(ny_merged_scaled, columns=geo_demo_rn)
ny_merged_scaled_df["kmeans_5_label"] = kmeans_5.labels_

#### Calculate descriptive statistics ----
k5means_s = ny_merged_scaled_df.groupby("kmeans_5_label")[geo_demo_rn].mean().round(2)

### Plot a radial plot ----
categories = k5means_s.columns
fi = go.Figure()
for g in k5means.index:
    fi.add_trace(
        go.Scatterpolar(
            r=k5means_s.loc[g].values,
            theta=categories,
            fill="toself",
            name=f"cluter #{g}",
        )
    )
fi.update_layout(
    polar={"radialaxis": {"visible": True, "range": [-2, 5]}},
    showlegend=True,
    title="KMeans Cluster Radial Plot",
    title_x=0.5,
)
fi.show()


# ------------------------------------------------------------------------------
#                      AGLOMERATIVE HIERARCHICAL CLUSTERS
# ------------------------------------------------------------------------------


## ---- Without Spatial Constraint ---------------------------------------------


### Set seed ----
np.random.seed(54321)

### Instantiate the AHC Algorithm ----
model = AgglomerativeClustering(n_clusters=5, linkage="ward")
model.fit(X=ny_merged_scaled)
ny_merge_2["ward5_label"] = model.labels_
ny_merged_scaled_df["ward5_label"] = model.labels_

### Profile ----
ward5sizes = ny_merge_2.groupby("ward5_label").size()

### Calculate descriptive statistics ----
ahc_nspat = ny_merged_scaled_df.groupby("ward5_label")[geo_demo_rn].mean().round(2)

### Radial plot ----
cat = ahc_nspat.columns
f = go.Figure()

for group in ahc_nspat.index:
    f.add_trace(
        go.Scatterpolar(
            r=ahc_nspat.loc[group].values,
            theta=cat,
            fill="toself",
            name=f"cluster {group}",
        )
    )

f.update_layout(
    polar={"radialaxis": {"visible": True, "range": [-2, 6]}},
    showlegend=True,
    title="AHC Cluster Radial Plot",
    title_x=0.5,
)
f.show()


## ---- With Spatial Constraints -----------------------------------------------


### Instantiate model ----
ahc_spat = AgglomerativeClustering(
    n_clusters=5, connectivity=w.sparse, linkage="ward"
).fit(X=ny_merged_scaled)

### Add model labels to the dataframe ----
ny_merge_2["ward5wgt_label"] = model.labels_
ny_merged_scaled_df["ward5wgt_label"] = model.labels_
ny_merge_2.plot(column="ward5wgt_label", legend=False, ax=ax)


### Use a KNN-distance matrix ----
knn_w = KNN.from_dataframe(ny_merge_2, k=10)

ahc_knn_spat = AgglomerativeClustering(
    n_clusters=5, connectivity=w.sparse, linkage="ward"
).fit(X=ny_merged_scaled)


ny_merge_2["ward5_knnwgt_label"] = model.labels_
ny_merged_scaled_df["ward5_knnwgt_label"] = model.labels_

### Plot ----
fig, ax = plt.subplots(ncols=1, figsize=(40, 20))
ny_merge_2.plot(
    column="ward5_knnwgt_label",
    legend=False,
    cmap="Set2",
    ax=ax,
    categorical=True,
    linewidth=0,
)
