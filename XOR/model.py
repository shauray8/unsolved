import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import trange
import numpy as np

## use batch loader, dont use numpy, optimize

class XOR(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(XOR, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        output, _ = self.lstm(input)
        output = self.fc(output[-1])
        return torch.sigmoid(output)

def create_dataset(size, variable=False):
    if variable:
        return torch.randint(2, size=(size, np.random.randint(1,50))).type(torch.float32)
    else:
        return torch.randint(2, size=(size,50)).type(torch.float32)

# Train the model
def training_loop(data, model, criterion, optimizer,Y):
    for epoch in ( a := trange(500)):
        running_loss = 0.0
        for X in range(len(data)):
            input_seq = data[X].unsqueeze(1).unsqueeze(2).to(torch.device("cuda"))
            optimizer.zero_grad()
            output = model(input_seq)
            loss = criterion(output[0], Y[X])
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        a.set_description('Epoch %d loss: %.3f' % (epoch + 1, running_loss / len(data)))


def someother():
# Generate a test set of binary strings
    X_test = np.random.randint(2, size=(10000, 50)).astype(np.float32)

# Compute parity for each sequence
    y_test = np.sum(X_test, axis=1) % 2
    y_test = torch.from_numpy(y_test).unsqueeze(1)

# Evaluate the model on the test set
    with torch.no_grad():
        correct = 0
        for i in range(len(X_test)):
            input_seq = torch.from_numpy(X_test[i]).unsqueeze(1).unsqueeze(2)
            output = model(input_seq)
            prediction = (output >= 0.5).float()
            correct += (prediction == y_test[i]).float().sum()

        accuracy = 100.0 * correct / len(X_test)
        print('Test accuracy: %.2f%%' % accuracy)


if __name__ == "__main__":
    data = create_dataset(100, True)

    Y = torch.sum(data, axis=1) % 2
    Y = Y.unsqueeze(1).to(torch.device("cuda"))

    model = XOR(input_size=1, hidden_size=32, output_size=1).to(torch.device("cuda"))
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())

    training_loop(data, model,criterion, optimizer,Y)
