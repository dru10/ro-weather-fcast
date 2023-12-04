import os
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv

from dataset import get_train_valid_dataset
from models import MyGRU, MyLSTM

BASE_PATH = os.path.join("/", *os.environ["VIRTUAL_ENV"].split("/")[:-1])

load_dotenv(os.path.join(BASE_PATH, "env", "train.env"))

# Dataset parameters
N_LAGS = int(os.getenv("N_LAGS"))

# LSTM parameters
HIDDEN_SIZE = int(os.getenv("HIDDEN_SIZE"))
NUM_LAYERS = int(os.getenv("NUM_LAYERS"))
DROPOUT = float(os.getenv("DROPOUT"))

# Train parameters
LEARNING_RATE = float(os.getenv("LEARNING_RATE"))
EPOCHS = int(os.getenv("EPOCHS"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model_path(model_type, hidden, lags, num_layers, epochs):
    return os.path.join(
        BASE_PATH,
        "models",
        str(model_type),
        "lags",
        str(lags),
        "hidden",
        str(hidden),
        "layers",
        str(num_layers),
        "dropout",
        str(DROPOUT),
        "learning_rate",
        str(LEARNING_RATE),
        "epochs",
        str(epochs),
    )


def build_set(ds, to_predict_idx, lags=10):
    train = []
    true = []
    for idx in range(len(ds) - lags):
        current = ds[idx : idx + lags]
        pred = ds[idx + lags][to_predict_idx]
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
        hidden = model.initialize_hidden(device)
        sample = 0
        for X, y_true in zip(x_train, y_train):
            sample += 1

            X = X.to(device)
            y_true = y_true.to(device)

            # Reset hidden state
            if type(hidden) == tuple:
                hidden = tuple([state.data for state in hidden])
            else:
                hidden = hidden.data

            optimizer.zero_grad()

            y_pred, hidden = model.forward(X, hidden)
            loss = criterion(y_pred, y_true)

            loss.backward()
            optimizer.step()

            if sample % 300 == 0:
                train_loss[epoch].append(loss.item())

                val_hidden = model.initialize_hidden(device)

                total_val_loss = 0

                model.eval()

                for X_val, y_val_true in zip(x_valid, y_valid):
                    X_val = X_val.to(device)
                    y_val_true = y_val_true.to(device)

                    if type(val_hidden) == tuple:
                        val_hidden = tuple(
                            [state.data for state in val_hidden]
                        )
                    else:
                        val_hidden = val_hidden.data

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
                    f"Avg Valid Loss: {avg_loss:3.8f}",
                )

    end = time.time()
    print(f"Duration = {end - start} seconds")

    return train_loss, valid_loss


def evaluate_model(model, x_valid, y_valid, temp_scaler):
    # Evaluate performance on validation set
    true_temperatures = temp_scaler.inverse_transform(y_valid)
    pred_temperatures = np.zeros(true_temperatures.shape)

    model.eval()
    hidden = model.initialize_hidden(device)
    for idx, x in enumerate(x_valid):
        x = x.to(device)

        if type(hidden) == tuple:
            hidden = tuple([state.data for state in hidden])
        else:
            hidden = hidden.data
        pred, hidden = model.forward(x, hidden)

        pred_temp = temp_scaler.inverse_transform(
            np.array(pred.item()).reshape(-1, 1)
        )
        pred_temperatures[idx] = pred_temp

    return true_temperatures, pred_temperatures


def plot_loss_curves(train_loss, valid_loss, model_path):
    tloss = [np.array(v).sum() / len(v) for k, v in train_loss.items()]
    vloss = [np.array(v).sum() / len(v) for k, v in valid_loss.items()]

    plt.figure()
    plt.plot(tloss, label="Training Loss")
    plt.plot(vloss, label="Validation Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Model training loss curves")
    plt.savefig(os.path.join(model_path, "loss_curves.jpg"))


def plot_validation_results(index, true, pred, model_path):
    plt.figure(figsize=(10, 4))
    plt.plot(index, true, label="True Temperature", color="#5884cc")
    plt.plot(index, pred, label="Predicted Temperature", color="#db3959")
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Average Temperature (°C)")
    plt.ylim((-10, 40))
    plt.title("Prediction results on validation set")
    plt.grid()
    plt.savefig(os.path.join(model_path, "validation_performance.jpg"))


if __name__ == "__main__":
    to_predict = "Temperature_Avg"
    datasets = get_train_valid_dataset(to_predict)

    to_predict_idx = datasets[to_predict]
    train_ds = datasets["train"]
    valid_ds = datasets["valid"]
    temp_scaler = datasets["temp_scaler"]
    valid_index = datasets["valid_timestamps"]

    for model_type in ["LSTM", "GRU"]:
        for hidden in [32, 64, 128]:
            for lags in [3, 5, 7]:
                x_train, y_train = build_set(train_ds, to_predict_idx, lags)

                # Add the last lags elements to valid_ds to ensure continuity
                x_valid, y_valid = build_set(
                    torch.cat((train_ds[-lags:], valid_ds)),
                    to_predict_idx,
                    lags,
                )

                for num_layers in [2, 4, 8]:
                    for epochs in [50, 100, 150]:
                        # Instantiate model
                        instance = MyLSTM if model_type == "LSTM" else MyGRU

                        model = instance(
                            input_size=train_ds.shape[1],
                            hidden_size=hidden,
                            num_layers=num_layers,
                            dropout=DROPOUT,
                        ).to(device)
                        criterion = nn.MSELoss()
                        optimizer = torch.optim.Adam(
                            model.parameters(), lr=LEARNING_RATE
                        )

                        train_loss, valid_loss = train_model(
                            model=model,
                            train_valid=(x_train, y_train, x_valid, y_valid),
                            criterion=criterion,
                            optimizer=optimizer,
                            epochs=epochs,
                        )

                        model_path = build_model_path(
                            model_type, hidden, lags, num_layers, epochs
                        )
                        os.makedirs(model_path, exist_ok=True)

                        plot_loss_curves(train_loss, valid_loss, model_path)

                        torch.save(
                            model.state_dict(),
                            os.path.join(model_path, "params.pt"),
                        )

                        true_temperatures, pred_temperatures = evaluate_model(
                            model=model,
                            x_valid=x_valid,
                            y_valid=y_valid,
                            temp_scaler=temp_scaler,
                        )

                        plot_validation_results(
                            valid_index,
                            true_temperatures,
                            pred_temperatures,
                            model_path,
                        )
