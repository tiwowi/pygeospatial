# ==============================================================================
#                 WORKING WITH COORDINATE REFERENCE SYSTEMS
# ==============================================================================


## ---- Load required libraries ------------------------------------------------


import geopandas as gpd
import matplotlib.pyplot as plt


## ---- Import shapefiles ------------------------------------------------------

url_world = (
    "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
)
url_capitals = (
    "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_populated_places.zip"
)
world = gpd.read_file(filename=url_world)
capitals = gpd.read_file(filename=url_capitals)

capitals = capitals[capitals["FEATURECLA"] == "Admin-0 capital"]

### Check the coordinate reference system ----
world.crs
capitals.crs

### Check if all shapefiles are in the same CRS ----
world.crs == capitals.crs


## ---- Plot a map that overlays shapefiles ------------------------------------

fig, ax = plt.subplots(figsize=(12, 10))
world.plot(ax=ax, color="lightgray")
capitals.plot(ax=ax, color="tomato", markersize=10, marker="*")
ax.set(xlabel="Longitude(Degrees)", ylabel="Latitude(Degrees)", title="WGS 1984 Datum")
plt.show()


## ---- Reprojecting the data --------------------------------------------------


### Azimuthal Equidistant ----

world_ae = world.to_crs(crs="ESRI:54032")
capitals_ae = capitals.to_crs(crs="ESRI:54032")

### Check the CRS ----
world_ae.crs

### Plot a map ----
fig, ax = plt.subplots(figsize=(15, 10))
world_ae.plot(ax=ax, color="blue", alpha=0.5)
capitals_ae.plot(ax=ax, color="red", marker="o", alpha=0.7)
plt.title("World Map in Azimuthal Equidistant Projection")
plt.show()


################################ End of workflow ###############################
