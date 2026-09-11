# ==============================================================================
#             CHAPTER 9 - DEVELOPING SPATIAL REGRESSION MODELS
# ==============================================================================


from pathlib import Path

import geopandas as gpd
import pandas as pd
import numpy as np
from pysal.model import spreg

## ---- Load pre-computed GeoDataFrame ------------------------------------------

### Loaded from data/ (produced by scripts/part_2/exploratory_data_visualization.py)
### instead of importing that script, which would re-run all of its code ----
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
listings_sub_gpd = gpd.read_parquet(DATA_DIR / "listings_sub_gpd.parquet")
boroughs = gpd.read_parquet(DATA_DIR / "boroughs.parquet")


### Subset Manhatan ----
manhatan = boroughs[boroughs["BoroName"] == "Manhattan"].to_crs("EPSG:4326")
listing_mask = listings_sub_gpd.geometry.within(manhatan.loc[3, "geometry"])
listing_manhatan = listings_sub_gpd[listing_mask]

### Select variables ----
voi = [
    "id",
    "property_type",
    "room_type",
    "beds",
    "price",
    "accommodates",
    "bedrooms",
    "review_scores_rating",
]

listing_manhatan_subset = listing_manhatan[voi].copy()

### Encode categorical variables ----
listing_manhatan_subset = pd.get_dummies(
    data=listing_manhatan_subset,
    columns=["room_type"],
    drop_first=True,
    sparse=True,
    dtype=int,
)

### Log the price variable ----
listing_manhatan_subset["log_price"] = np.log(listing_manhatan_subset["price"])


### Fit a model ----
#### Define a list of explanatory variables ----
explanatory_vars = [
    "beds",
    "room_type_Hotel room",
    "room_type_Private room",
    "room_type_Shared room",
    "accommodates",
    "bedrooms",
    "review_scores_rating",
]

#### Model ----
ols_model = spreg.OLS(
    y=listing_manhatan_subset["log_price"].values,
    x=listing_manhatan_subset[explanatory_vars].values,
    name_y="price",
    name_x=explanatory_vars,
)
