# ==============================================================================
#                         READING IN GEO FILES
# ==============================================================================

## ---- Load required libraries ------------------------------------------------


import geopandas as gpd
import matplotlib.pyplot as plt


## ---- Import input dataset from upstream workflow ----------------------------


import sys
from scripts.part_1.working_with_crs import world, capitals, url_capitals, url_world


## ---- Leveraging `mask` argument ---------------------------------------------


capitals_labelrank = gpd.read_file(
    filename=url_capitals, mask=capitals[capitals["LABELRANK"] == 3]
)


## ---- Leveraging `bbox` parameter --------------------------------------------


bbox = (180 - 16.06713, 180 - 16.55522)  # This needs a length-four tuple
world = gpd.read_file(filename=url_world, bbox=bbox)


## ---- Filtering rows ---------------------------------------------------------


gpd.read_file(
    filename=url_world, rows=slice(1, 4)
)  # returns the first three rows. If I use an integer (20), it
# will return the first 20 rows.


## ---- Read in a file with selected columns besides geometry ------------------


gpd.read_file(filename=url_world, columns=["NAME", "ISO_A3"])


## ---- Leveraging `ignore_geometry` to read in file without it ----------------


gpd.read_file(filename=url_world, columns=["NAME", "ISO_A3"], ignore_geometry=True)
