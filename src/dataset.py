import os

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler

BASE_PATH = os.path.join("/", *os.environ["VIRTUAL_ENV"].split("/")[:-1])

load_dotenv(os.path.join(BASE_PATH, "env", "data.env"))
load_dotenv(os.path.join(BASE_PATH, "env", "dataset.env"))

weather_station = os.getenv("WEATHER_STATION")
start_year = int(os.getenv("START_YEAR"))
end_year = int(os.getenv("END_YEAR"))

dataset_path = os.path.join(
    BASE_PATH, "datasets", f"{weather_station}_{start_year}_{end_year}.csv"
)


def clean_data(df):
    # Set the date as the index
    index = pd.to_datetime(df[["Year", "Month", "Day"]])
    df.set_index(index, inplace=True)

    # Clean rows
    # Sea Level Pressure, gets dropped, too many 0, not enough relevant info
    corrupted = df[df["Sea_Level_Pressure"] == "True"]
    df.drop(index=corrupted.index, inplace=True)
    df.drop("Sea_Level_Pressure", axis=1, inplace=True)

    # Humidity Average, 11 rows with NaN, can drop them
    corrupted = df[df["Humidity_Avg"].isna()]
    df.drop(index=corrupted.index, inplace=True)

    # Precipitation Total, fill Nan with 0
    df["Precipitation_Total"].fillna("0", inplace=True)

    # Visibility Average, gets dropped, too many NaN values > 10%
    df.drop("Visibility_Avg", axis=1, inplace=True)

    # Wind Max, contains mostly NaN values, can drop
    # We have Wind Sustained Max which is clean
    df.drop("Wind_Max", axis=1, inplace=True)

    return df


def normalize_temp_data(df, temp_cols):
    # Create new dataframe
    scaled = df.copy(deep=True)

    mintemp = int(os.getenv("MIN_TEMP"))
    maxtemp = int(os.getenv("MAX_TEMP"))

    # Scale with record min and max
    temp_scaler = MinMaxScaler()
    temp_scaler.fit(np.array((mintemp, maxtemp)).reshape(-1, 1))

    cols = temp_cols
    temp_data = np.array(df[cols]).T
    scaled = df.copy(deep=True)
    temp_scaled = [temp_scaler.transform(X.reshape(-1, 1)) for X in temp_data]
    for idx, col in enumerate(cols):
        scaled[col] = temp_scaled[idx]
    return scaled, temp_scaler


def normalize_data(df, cols):
    scaler = MinMaxScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df, scaler


def get_train_valid_dataset(to_predict):
    df = pd.read_csv(dataset_path)
    clean_data(df)

    # Boolean columns
    cat_cols = [col for col in df.columns if col.startswith("Is")]
    # Continous columns
    cont_cols = list(
        set(df.columns) - set(cat_cols) - set(["Year", "Month", "Day"])
    )
    # Temperature columns
    temp_cols = [col for col in cont_cols if col.startswith("Temperature")]
    # Miscellaneous continuous columns -> Humidity, Precipitation, Wind
    misc_cont_cols = list(set(cont_cols) - set(temp_cols))

    # Set correct datatypes
    df[cont_cols] = df[cont_cols].astype("float")

    # Scale temperatures on the whole dataset
    scaled, temp_scaler = normalize_temp_data(df, temp_cols)

    valid_start = f"{end_year}-01-01"
    train_df = scaled[:valid_start][:-1]
    valid_df = scaled[valid_start:]

    # Scale the rest of the continous columns
    train_df, scaler = normalize_data(train_df, misc_cont_cols)
    valid_df[misc_cont_cols] = scaler.transform(valid_df[misc_cont_cols])

    # Convert to tensors
    train_cont = torch.tensor(
        np.stack([train_df[cont_cols]]), dtype=torch.float
    )
    train_cat = torch.tensor(np.stack([train_df[cat_cols]]), dtype=torch.float)
    valid_cont = torch.tensor(
        np.stack([valid_df[cont_cols]]), dtype=torch.float
    )
    valid_cat = torch.tensor(np.stack([valid_df[cat_cols]]), dtype=torch.float)

    # Create a train and valid dataset
    train_ds = torch.cat((train_cont, train_cat), dim=2).squeeze()
    valid_ds = torch.cat((valid_cont, valid_cat), dim=2).squeeze()

    to_predict_idx = cont_cols.index(to_predict)

    return {to_predict: to_predict_idx, "train": train_ds, "valid": valid_ds}
