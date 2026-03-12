import pandas as pd

air_quality = pd.read_csv("data/air_quality_no2.csv", index_col=0, parse_dates=True)

# print(air_quality.head())

air_quality["london_mg_per_cubic"] = air_quality["station_london"] * 1.882

# print(air_quality.head()) # temperature of 25 degrees Celsius and pressure of 1013 hPa, the conversion factor is 1.882

air_quality["ratio_paris_antwerp"] = ( air_quality["station_paris"] / air_quality["station_antwerp"] )
# check the ratio of the values in Paris versus Antwerp and save the result in a new column

# print(air_quality.head())

air_quality_renamed  = air_quality.rename( # rename the data columns to the corresponding station identifiers used by OpenAQ
    columns = {
        "station_antwerp": "BETR801",
        "station_paris": "FR04014",
        "station_london": "London Westminster",
    }
)

# print(air_quality_renamed.head())

# converting the column names to lowercase letters

air_quality_renamed = air_quality_renamed.rename(columns=str.lower)

# print(air_quality_renamed.head())
