# ==============================================================================
#             CHAPTER 8 - SPATIAL CLUSTERING  AND REGIONALISATION
# ==============================================================================


# ------------------------------------------------------------------------------
#                      GET SAMPLE DATA FROM THE US CENSUS
# ------------------------------------------------------------------------------


## ---- Load required libraries ------------------------------------------------


from census import Census
from us import states
import geopandas as gpd
import pandas as pd


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
