import os
import csv

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv

from dataset import clean_data
from models import MyGRU, MyLSTM
from train import build_set, plot_loss_curves, train_model

BASE_PATH = os.path.join("/", *os.environ["VIRTUAL_ENV"].split("/")[:-1])

load_dotenv(os.path.join(BASE_PATH, "env", "future.env"))

MODEL_TYPE = os.getenv("MODEL_TYPE")
INDEX = int(os.getenv("INDEX"))

LAGS = int(os.getenv("LAGS"))
HIDDEN = int(os.getenv("HIDDEN"))
LAYERS = int(os.getenv("LAYERS"))
DROPOUT = float(os.getenv("DROPOUT"))
LEARNIG_RATE = float(os.getenv("LEARNING_RATE"))
EPOCHS = int(os.getenv("EPOCHS"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_train_test_sets(train_df, test_df, scaler):
    train_df = clean_data(train_df)
    test_df = clean_data(test_df)

    train_df = train_df["Temperature_Avg"]
    test_df = test_df["Temperature_Avg"]

    train_scaled = scaler.transform(np.array(train_df).reshape(-1, 1))
    test_scaled = scaler.transform(np.array(test_df).reshape(-1, 1))

    train_set = torch.tensor(train_scaled, dtype=torch.float32)
    test_set = torch.tensor(test_scaled, dtype=torch.float32)

    x_train, y_train = build_set(train_set, 0, LAGS)
    x_test, y_test = build_set(
        torch.cat((train_set[-LAGS:], test_set)), 0, LAGS
    )

    return x_train, y_train, x_test, y_test


def make_future_predictions(model, start, steps, scaler):
    model.eval()

    pred_temps = []
    hidden = model.initialize_hidden(device)

    for i in range(steps):
        start = start.to(device)
        if type(hidden) == tuple:
            hidden = tuple([state.data for state in hidden])
        else:
            hidden = hidden.data

        out, hidden = model(start, hidden)
        pred_temp = float(
            scaler.inverse_transform(np.array(out.item()).reshape(-1, 1))
        )
        pred_temps.append(pred_temp)

        start = torch.cat((start.flatten(), out), dim=0)[1:].reshape(-1, 1)

    return pred_temps


def plot_prediction_performance(index, true_temps, pred_temps, model_path):
    plt.figure()
    plt.plot(index, pred_temps, label="Predicted")
    plt.plot(index, true_temps, label="True")
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Average Temperature (°C)")
    plt.title("Extended future prediction results")
    plt.savefig(os.path.join(model_path, "future_prediction.jpg"))


def save_results(csv_path, pred_temps, true_temps):
    rmse = np.sqrt(np.mean((pred_temps - true_temps)) ** 2)
    mape = np.mean(np.abs((true_temps - pred_temps) / true_temps)) * 100
    R, p = pearsonr(true_temps, pred_temps)

    headers = ["MODEL_TYPE", "INDEX", "RMSE", "MAPE", "R"]
    stats = [MODEL_TYPE, str(INDEX), rmse, mape, R]
    with open(csv_path, "a", newline="") as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            # File is empty, write headers
            writer.writerow(headers)
        writer.writerow(stats)


train_df = pd.read_csv(os.getenv("TRAIN_DS_PATH"))
test_df = pd.read_csv(os.getenv("TEST_DS_PATH"))

instance = MyLSTM if MODEL_TYPE == "LSTM" else MyGRU

scaler = MinMaxScaler()
scaler.fit(np.array((-35, 45)).reshape(-1, 1))

x_train, y_train, x_test, y_test = load_train_test_sets(
    train_df, test_df, scaler
)

model = instance(
    input_size=x_train.shape[2],
    hidden_size=HIDDEN,
    num_layers=LAYERS,
    dropout=DROPOUT,
).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNIG_RATE)

train_loss, test_loss = train_model(
    model=model,
    train_valid=(x_train, y_train, x_test, y_test),
    criterion=criterion,
    optimizer=optimizer,
    epochs=EPOCHS,
)

model_path = os.path.join("models", "future", MODEL_TYPE, str(INDEX))
os.makedirs(model_path, exist_ok=True)

torch.save(model.state_dict(), os.path.join(model_path, "params.pt"))
plot_loss_curves(train_loss, test_loss, model_path)

# model.load_state_dict(torch.load(os.path.join(model_path, "params.pt")))

steps = 10
start = x_test[0]
pred_temps = make_future_predictions(model, start, steps, scaler)

true_temps = np.array(test_df[:steps]["Temperature_Avg"], dtype=np.float32)
index = test_df[:steps].index
plot_prediction_performance(index, true_temps, pred_temps, model_path)

save_results("future.csv", pred_temps, true_temps)
