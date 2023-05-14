import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
from tqdm import trange

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

X_train = np.random.randint(2, size=(1000, 50)).astype(np.float32)

# Compute parity for each sequence
y_train = np.sum(X_train, axis=1) % 2
y_train = torch.from_numpy(y_train).unsqueeze(1)

# Define the model, loss function, and optimizer
model = XOR(input_size=1, hidden_size=32, output_size=1).to(torch.device("cuda"))
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

# Train the model
for epoch in trange(10):
    running_loss = 0.0
    for i in range(len(X_train)):
        optimizer.zero_grad()
        input_seq = torch.from_numpy(X_train[i]).unsqueeze(1).unsqueeze(2).to(torch.device("cuda"))
        output = model(input_seq)
        loss = criterion(output[0], y_train[i].to(torch.device("cuda")))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print('Epoch %d loss: %.3f' % (epoch + 1, running_loss / len(X_train)))

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

