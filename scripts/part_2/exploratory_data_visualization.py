# ==============================================================================
#                 CHAPTER 5 - EXPLORATORY DATA VISUALIZATION
# ==============================================================================


## ---- Load required libraries ------------------------------------------------


# Standard library
import statistics

import geopandas as gpd
import geoplot as gplt
import geoplot.crs as gcrs
import geoviews
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

geoviews.extension("bokeh")

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


## -------------------------------------- Exploratory Spatial Data Analysis ----


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


## ---- Converting point data up to higher-order geographies -------------------


### Read in the New York Census Tracts ----
ny_tracts_path = (
    "https://www2.census.gov/geo/tiger/TIGER2021/TRACT/tl_2021_36_tract.zip"
)
ny_tracts = gpd.read_file(ny_tracts_path)
ny_tracts.to_crs(4326, inplace=True)

### Subset the census tracts to those in the New York CBSA ----
cbsa_path = "https://www2.census.gov/geo/tiger/TIGER2021/CBSA/tl_2021_us_cbsa.zip"
cbsas = gpd.read_file(cbsa_path)
ny_cbsa = cbsas[cbsas["GEOID"] == "35620"]
mask = ny_tracts.intersects(ny_cbsa.loc[620, "geometry"]).loc()

### Aggregate the airbnb locations to the NY census tracts ----
ny_tracts_sj = gpd.sjoin(left_df=ny_tracts, right_df=listings_sub_gpd, how="left")
ny_tracts_sj = ny_tracts_sj[["GEOID", "price", "geometry"]]
ny_tracts_agg = ny_tracts_sj.dissolve(by="GEOID", aggfunc="mean")

### Visualise the distribution of price across census tract ----
gplt.choropleth(
    ny_tracts_agg,
    hue="price",
    cmap="inferno_r",
    legend=True,
    figsize=(60, 15),
    legend_kwargs={"orientation": "vertical"},
)

### Exclude outliers ----
#### Get mean and standard deviation of price ----
mean_price = statistics.mean(ny_tracts_agg["price"].dropna())
stdev = statistics.stdev(ny_tracts_agg["price"].dropna())

#### Drop records that are outliers ----
ny_tracts_agg = ny_tracts_agg[ny_tracts_agg["price"] < mean_price + stdev]

#### Plot ----
gplt.choropleth(
    ny_tracts_agg,
    hue="price",
    cmap="inferno_r",
    legend=True,
    figsize=(60, 15),
    legend_kwargs={"orientation": "vertical"},
)

#### Plot with geoviews ----
map = geoviews.Polygons(data=ny_tracts_agg, vdims=["price", "GEOID"]).opts(
    height=600,
    width=900,
    title="NYC Tract Price Distribution",
    tools=["hover", "wheel_zoom", "box_select"],
    cmap="viridis",
    colorbar=True,
    colorbar_position="bottom",
)
