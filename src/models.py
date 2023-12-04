import torch.nn as nn
import torch


class BaseRNN(nn.Module):
    def __init__(
        self,
        hidden_size=128,
        num_layers=4,
        output_size=1,
        dropout=0.5,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, input, hidden):
        # Does the fully connected oprations after the RNN operations
        # implemented in children
        out = self.dropout(input)
        out = self.fc(out)
        # Only care about last prediction
        return out[-1], hidden


class MyLSTM(BaseRNN):
    def __init__(
        self,
        input_size,
        hidden_size=128,
        num_layers=4,
        output_size=1,
        dropout=0.5,
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            dropout=dropout,
        )

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(self, X, hidden):
        out, hidden = self.lstm(X, hidden)
        return super().forward(out, hidden)

    def initialize_hidden(self, device):
        hidden = (
            torch.zeros(self.num_layers, self.hidden_size).to(device),
            torch.zeros(self.num_layers, self.hidden_size).to(device),
        )
        return hidden


class MyGRU(BaseRNN):
    def __init__(
        self,
        input_size,
        hidden_size=128,
        num_layers=4,
        output_size=1,
        dropout=0.5,
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            dropout=dropout,
        )

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(self, X, hidden):
        out, hidden = self.gru(X, hidden)
        return super().forward(out, hidden)

    def initialize_hidden(self, device):
        hidden = torch.zeros(self.num_layers, self.hidden_size).to(device)
        return hidden
