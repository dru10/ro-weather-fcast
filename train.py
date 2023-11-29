import os
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

# Bucharest record min and max temperatures, adjusted according to source
# Source: https://www.extremeweatherwatch.com/cities/bucharest/lowest-temperatures
BUC_MIN_TEMP = -35
BUC_MAX_TEMP = 45

# Dataset parameters
N_LAGS = 7

# LSTM parameters
HIDDEN_SIZE = 128
NUM_LAYERS = 4
DROPOUT = 0.5

# Train parameters
LEARNING_RATE = 0.001
EPOCHS = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MyLSTM(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size=128,
        num_layers=4,
        output_size=1,
        dropout=0.5,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, X, hidden):
        out, hidden = self.lstm(X, hidden)
        out = self.dropout(out)
        out = self.fc(out)
        # Only care about last prediction
        return out[-1], hidden

    def initialize_hidden(self):
        hidden = (
            torch.zeros(self.num_layers, self.hidden_size).to(device),
            torch.zeros(self.num_layers, self.hidden_size).to(device),
        )
        return hidden


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


def normalize_temp_data(df):
    # Create new dataframe
    scaled = df.copy(deep=True)

    # Scale with record min and max
    temp_scaler = MinMaxScaler()
    temp_scaler.fit(np.array((BUC_MIN_TEMP, BUC_MAX_TEMP)).reshape(-1, 1))

    cols = temp_cols
    temp_data = np.array(df[cols]).T
    scaled = df.copy(deep=True)
    temp_scaled = [temp_scaler.transform(X.reshape(-1, 1)) for X in temp_data]
    for idx, col in enumerate(cols):
        scaled[col] = temp_scaled[idx]
    return scaled, temp_scaler


def normalize_data(df):
    scaler = MinMaxScaler()
    cols = misc_cont_cols
    df[cols] = scaler.fit_transform(df[cols])
    return df, scaler


def build_set(ds, lags=10):
    train = []
    true = []
    for idx in range(len(ds) - lags):
        current = ds[idx : idx + lags]
        pred = ds[idx + lags][temp_avg_idx]
        train.append(current)
        true.append(pred)

    train = torch.cat([t for t in train]).reshape(-1, lags, ds.shape[1])
    true = torch.tensor(true).reshape(-1, 1)
    return train, true


def train_model(model, train_valid: tuple, criterion, optimizer, epochs=50):
    start = time.time()

    train_loss = defaultdict(list)
    valid_loss = defaultdict(list)

    x_train, y_train, x_valid, y_valid = train_valid

    for epoch in range(epochs):
        hidden = model.initialize_hidden()
        sample = 0
        for X, y_true in zip(x_train, y_train):
            sample += 1

            X = X.to(device)
            y_true = y_true.to(device)

            # Reset hidden state
            hidden = tuple([state.data for state in hidden])

            optimizer.zero_grad()

            y_pred, hidden = model.forward(X, hidden)
            loss = criterion(y_pred, y_true)

            loss.backward()
            optimizer.step()

            if sample % 300 == 0:
                train_loss[epoch].append(loss.item())

                val_hidden = model.initialize_hidden()

                total_val_loss = 0

                model.eval()

                for X_val, y_val_true in zip(x_valid, y_valid):
                    X_val = X_val.to(device)
                    y_val_true = y_val_true.to(device)

                    val_hidden = tuple([state.data for state in val_hidden])

                    y_val_pred, val_hidden = model.forward(X_val, val_hidden)
                    val_loss = criterion(y_val_pred, y_val_true)

                    total_val_loss += val_loss.item()

                avg_loss = total_val_loss / len(x_valid)
                valid_loss[epoch].append(avg_loss)

                model.train()

                print(
                    f"Epoch: {epoch:{len(str(epochs))}}",
                    f"Sample: {sample:{len(str(x_train.shape[0]))}}",
                    f"Train Loss: {loss.item():3.8f}",
                    f"Avg Valid Loss: {avg_loss}",
                )

    end = time.time()
    print(f"Duration = {end - start} seconds")

    return train_loss, valid_loss


def plot_loss_curves(train_loss, valid_loss):
    tloss = [np.array(v).sum() / len(v) for k, v in train_loss.items()]
    vloss = [np.array(v).sum() / len(v) for k, v in valid_loss.items()]

    plt.figure()
    plt.plot(tloss, label="Training Loss")
    plt.plot(vloss, label="Validation Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Model training loss curves")
    plt.savefig("plots/loss_curves.jpg")


df = pd.read_csv("data.csv")
df = clean_data(df)

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
scaled, temp_scaler = normalize_temp_data(df)

valid_start = "2022-01-01"
train_df = scaled[:valid_start][:-1]
valid_df = scaled[valid_start:]

# Scale the rest of the continous columns
train_df, scaler = normalize_data(train_df)
valid_df[misc_cont_cols] = scaler.transform(valid_df[misc_cont_cols])

# Convert to tensors
train_cont = torch.tensor(np.stack([train_df[cont_cols]]), dtype=torch.float)
train_cat = torch.tensor(np.stack([train_df[cat_cols]]), dtype=torch.float)
valid_cont = torch.tensor(np.stack([valid_df[cont_cols]]), dtype=torch.float)
valid_cat = torch.tensor(np.stack([valid_df[cat_cols]]), dtype=torch.float)

# Create a train and valid dataset
train_ds = torch.cat((train_cont, train_cat), dim=2).squeeze()
valid_ds = torch.cat((valid_cont, valid_cat), dim=2).squeeze()

temp_avg_idx = cont_cols.index("Temperature_Avg")

x_train, y_train = build_set(train_ds, N_LAGS)

# Add the last N_LAGS elements to valid_ds to ensure continuity
x_valid, y_valid = build_set(torch.cat((train_ds[-N_LAGS:], valid_ds)), N_LAGS)

# Instantiate model
model = MyLSTM(
    input_size=train_ds.shape[1],
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_loss, valid_loss = train_model(
    model=model,
    train_valid=(x_train, y_train, x_valid, y_valid),
    criterion=criterion,
    optimizer=optimizer,
    epochs=EPOCHS,
)

plot_loss_curves(train_loss, valid_loss)


model_name = f"LSTM_{N_LAGS}lags_{HIDDEN_SIZE}hidden_{NUM_LAYERS}layers_{DROPOUT}dropout_{LEARNING_RATE}lr_{EPOCHS}epochs.model"
torch.save(model, os.path.join("models", model_name))
