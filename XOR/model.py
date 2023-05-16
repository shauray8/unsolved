
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import trange
import os
import pickle

import torch.backends.cudnn as cudnn
import numpy as np
from torch.utils.tensorboard import SummaryWriter

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


def save_checkpoint(state, save_path, filename='variable_checkpoint.pkl'):
    with open(os.path.join(save_path, filename), 'wb') as handle:
        pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Train the model
def training_loop(data, model, criterion, optimizer,Y, i, train_writter, test_writter):
    for epoch in ( a := trange(100)):
        running_loss = 0.0
        for X in range(len(data)):
            input_seq = data[X].unsqueeze(1).unsqueeze(2).to(torch.device("cuda"))
            optimizer.zero_grad()
            output = model(input_seq)
            loss = criterion(output[0], Y[X])
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        acc = test(model, criterion, optimizer, epoch)
        train_writer.add_scalar('TRAINING lOSS', loss.item(), epoch)

        if epoch % 2:
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.module.state_dict(),
            }, "./", f"Checkpoint_{i}.pkl")

        a.set_description('Epoch %d loss: %.4f ; val acc : %.4f' % (epoch + 1, running_loss / len(data), acc))

def test(model, criterion, optimizer, epoch):
    X_test = create_dataset(10,True)
    y_test = torch.sum(X_test, axis=1) % 2
    y_test = y_test.unsqueeze(1)

    with torch.no_grad():
        correct = 0
        for i in range(len(X_test)):
            input_seq = X_test[i].unsqueeze(1).unsqueeze(2).to(torch.device("cuda"))
            output = model(input_seq)
            prediction = (output >= 0.5).float()
            correct += (prediction == y_test[i].to(torch.device("cuda"))).float().sum()

        accuracy = 100.0 * correct / len(X_test)
        val_writer.add_scalar('VALIDATION lOSS', accuracy, epoch)
    return accuracy

if __name__ == "__main__":
    torch.cuda.empty_cache()

    model = XOR(input_size=1, hidden_size=32, output_size=1).to(torch.device("cuda"))
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr = .0005)

    for i in range(0,2):
        train_writer = SummaryWriter(os.path.join("./", f"train{i}"))
        val_writer = SummaryWriter(os.path.join("./", f"validate{i}"))
        data = create_dataset(1000, i)

        Y = torch.sum(data, axis=1) % 2
        Y = Y.unsqueeze(1).to(torch.device("cuda"))

        training_loop(data, model,criterion, optimizer,Y, i, train_writer, val_writer)
