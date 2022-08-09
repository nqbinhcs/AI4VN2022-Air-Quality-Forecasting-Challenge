import torch
import torch.nn as nn
from torch.autograd import Variable

from models.base import BaseModel


class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))

        h_out = h_out.view(-1, self.hidden_size)

        out = self.fc(h_out)

        return out


class LSTMmodel(BaseModel):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length, num_epochs=2000, learning_rate=0.01):
        super(LSTMmodel, self).__init__()

        self.model = LSTM(num_classes, input_size,
                          hidden_size, num_layers, seq_length)

        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate)

    def get_valid(self, X, y):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
            loss = self.criterion(outputs, y)
            return loss.item()

    def fit(self, X, y, eval_set, early_stopping_rounds):

        X_train = X
        y_train = y
        X_valid, y_valid = eval_set

        for epoch in range(self.num_epochs):
            self.model.train()

            outputs = self.model(X_train)
            self.optimizer.zero_grad()

            # obtain the loss function
            loss = self.criterion(outputs, y_train)

            loss.backward()

            self.optimizer.step()

            # print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

            print(
                f'Epoch {epoch + 1}/{self.num_epochs}, training loss = {loss.item()}, validation loss = {self.get_valid(X_valid, y_valid)}')
