# ==============================================================================
#                 CHAPTER 5 - EXPLORATORY DATA VISUALIZATION
# ==============================================================================


## ---- Load required libraries ------------------------------------------------


import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import pysal
import splot
import seaborn as sns
import geoplot.crs as gcrs
import geoplot as gplt


## ---- Read in data -----------------------------------------------------------


inside_airbnb = "https://data.insideairbnb.com/united-states/ny/new-york-city/2026-08-10/data/listings.csv.gz"
data_dictionary = "https://docs.google.com/spreadsheets/d/1iWCNJcSutYqpULSQHlNyGInUvHg2BoUGoNRIGa6Szc4/edit?gid=1322284596#gid=1322284596"
listings = pd.read_csv(inside_airbnb)

### Subset required columns only ----
vars = [
    "id",
    "property_type",
    "neighbourhood_cleansed",
    "neighbourhood_group_cleansed",
    "beds",
    "bathrooms",
    "price",
    "latitude",
    "longitude",
]

listings_sub = listings[vars]

### Drop variables with missing values ----
listings_sub = (
    listings_sub.drop(columns=["bathrooms"])
    .assign(price=lambda x: x["price"].replace("[$,]", "", regex=True).astype(float))
    .dropna(axis=0)
)

### Visualise the distribution of New York City's Airbnb nightly prices ----
sns.displot(listings_sub["price"], kde=True)
plt.show()


## ---- Exploratory Spatial Data Analysis --------------------------------------


### Convert pandas dataframe to geopandas dataframe ----
listings_sub_gpd = gpd.GeoDataFrame(
    listings_sub,
    geometry=gpd.points_from_xy(listings_sub["longitude"], listings_sub["latitude"]),
    crs=4326,
)

### Plot a map ----

#### Point Plot ----
gplt.pointplot(
    df=listings_sub_gpd,
    hue="beds",
    legend=True,
    projection=gcrs.Mercator(),
    cmap="viridis",
)

#### Polyplot and Kernel-density plot ----
boroughs = gpd.read_file(gplt.datasets.get_path("nyc_boroughs")).to_crs(4326)
boroughs.plot()

ax = gplt.kdeplot(
    df=listings_sub_gpd,
    cmap="inferno_r",
    fill=True,
    clip=boroughs.geometry,
    projection=gcrs.WebMercator(),
)
gplt.polyplot(df=boroughs, ax=ax, zorder=1)


#### Choropleth map -----
contiguous_usa = gpd.read_file(gplt.datasets.get_path("contiguous_usa"))
gplt.choropleth(
    df=contiguous_usa,
    hue="population",
    cmap="Reds",
    legend=True,
    legend_kwargs={"orientation": "horizontal"},
)
