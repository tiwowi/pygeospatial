# ==============================================================================
#            CHAPTER 6 - HYPOTHESIS TESTING AND SPATIAL RANDOMNESS
# ==============================================================================


## ---- Load required libraries ------------------------------------------------


from pandas.core.interchange import column
import contextily  # noqa: F401
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn
import seaborn as sns
import splot
from splot import esda as esdaplot
from IPython.display import Markdown, display, display_html, display_markdown
from numpy.random import seed
from pysal.explore import esda
from pysal.lib import weights
from splot.esda import plot_moran

# ------------------------------------------------------------------------------
#                      GLOBAL SPATIAL AUTOCORRELATION
# ------------------------------------------------------------------------------
## ---- Read in data -----------------------------------------------------------
from scripts.part_2.exploratory_data_visualization import ny_tracts_agg

## ---- Calculate Spatial Weights ----------------------------------------------


### Drop missing values ----
ny_tracts_agg.dropna(inplace=True)
w = weights.Queen.from_dataframe(ny_tracts_agg)

### Row standardize ----
w.transform = "R"

### Calculate spatial lag ----
spatial_lag = ny_tracts_agg["price_lag"] = weights.spatial_lag.lag_spatial(
    w=w, y=ny_tracts_agg["price"]
)

### Standardize price and the lag price ----
ny_tracts_agg = ny_tracts_agg.assign(
    price_std=ny_tracts_agg["price"] - ny_tracts_agg["price"].mean(),
    price_lag_std=ny_tracts_agg["price_lag"] - ny_tracts_agg["price_lag"].mean(),
)


## ---- Moran's I Statistic ----------------------------------------------------


### Plot Moran's I scatterplot ----
fig, ax = plt.subplots(1, figsize=(18, 10))
sns.regplot(
    data=ny_tracts_agg,
    x="price_std",
    y="price_lag_std",
    ci=None,
    line_kws={"color": "r"},
)
ax.axvline(0, color="k", alpha=0.8)
ax.axhline(0, color="k", alpha=0.8)
ax.set_title("Moran's I Plot - NYC Airbnb Price")
ax.set_xlabel("Standardized Price")
ax.set_ylabel("Standardized Price Lag")
plt.show()


morans_stat = esda.moran.Moran(y=ny_tracts_agg["price"], w=w)
display(Markdown(f"""**Moran's I:** {morans_stat.I}"""))
display(Markdown(f"""**p-value:** {morans_stat.p_sim}"""))

### Plot Moran's I leveraging `plot_moran` function ----
plot_moran(morans_stat)


## ---- Geary's C statistic ----------------------------------------------------


geary_c = esda.geary.Geary(y=ny_tracts_agg.price, w=w)
display(Markdown(f"""**Geary's C:** {geary_c.C:.3f}"""))
display(Markdown(f"""**p-valu:** {geary_c.p_sim}"""))


# ------------------------------------------------------------------------------
#                      LOCAL SPATIAL AUTOCORRELATION
# ------------------------------------------------------------------------------


## ---- Local Moran's Index ----------------------------------------------------


### Calculate Local Moran's Index ----
price_lisa = esda.moran.Moran_Local(ny_tracts_agg.price, w)

### Plot ----
f, ax = plt.subplots(ncols=1, figsize=(10, 8))
sns.kdeplot(price_lisa.Is, ax=ax)

### Visualise results ----
#### Add LISA's to Geopandas dataframe -----
ny_tracts_agg["ML_Is"] = price_lisa.Is

#### Create a LISA-cluster map ----
fig, ax = plt.subplots(ncols=1, figsize=(10, 10))
ny_tracts_agg.plot(
    column="ML_Is",
    cmap="vlag",
    scheme="quantiles",
    k=4,
    edgecolor="white",
    linewidth=0.1,
    alpha=0.75,
    legend=True,
    ax=ax,
)
plt.show()

#### Create a LISA-cluster map: HH, HL, LH, LL ----
fix, ax = plt.subplots(ncols=1, figsize=(10, 10))
esdaplot.lisa_cluster(moran_loc=price_lisa, gdf=ny_tracts_agg, ax=ax)
