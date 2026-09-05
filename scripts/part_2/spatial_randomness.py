# ==============================================================================
#            CHAPTER 6 - HYPOTHESIS TESTING AND SPATIAL RANDOMNESS
# ==============================================================================


## ---- Load required libraries ------------------------------------------------


import seaborn
import contextily
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import splot
from pysal.lib import weights
from pysal.explore import esda
from splot.esda import plot_moran
from numpy.random import seed
from IPython.display import (
    display,
    Markdown,
    display_latex,
    display_markdown,
    display_html,
)

## ---- Read in data -----------------------------------------------------------

import sys
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


## ---- Calculate Moran's I Statistic ------------------------------------------


morans_stat = esda.moran.Moran(y=ny_tracts_agg["price"], w=w)
display(Markdown(f"""**Moran's I:** {morans_stat.I}"""))
display(Markdown(f"""**p-value:** {morans_stat.p_sim}"""))

### Plot Moran's I leveraging `plot_moran` function ----
plot_moran(morans_stat)


## ---- Testing for Spatial Randomness leveraging Geary's C statistic ----


geary_c = esda.geary.Geary(y=ny_tracts_agg.price, w=w)
display(Markdown(f"""**Geary's C:** {geary_c.C:.3f}"""))
display(Markdown(f"""**p-valu:** {geary_c.p_sim}"""))
