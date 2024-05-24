import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler

import time
import numpy as np

from dataset import Shakespeare
from model import CharRNN, CharLSTM

def train(model, trn_loader, device, criterion, optimizer):
    model.train()
    epoch_loss = 0
    for inputs, targets in trn_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        hidden = model.init_hidden(inputs.size(0), device)
        optimizer.zero_grad()
        
        outputs, hidden = model(inputs, hidden)
        loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(trn_loader)

def validate(model, val_loader, device, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            hidden = model.init_hidden(inputs.size(0), device)
            outputs, hidden = model(inputs, hidden)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            epoch_loss += loss.item()
            
    return epoch_loss / len(val_loader)

def main():
    input_file = '../data/shakespeare_train.txt'  # 데이터셋 경로
    batch_size = 128
    hidden_size = 256
    num_layers = 2
    learning_rate = 0.001
    num_epochs = 50

    dataset = Shakespeare(input_file)
    n = len(dataset)
    indices = list(range(n))
    split = int(0.2 * n)
    np.random.seed(int(time.time()))
    np.random.shuffle(indices)
    trn_indices, val_indices = indices[split:], indices[:split]
    trn_sampler = SubsetRandomSampler(trn_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    trn_loader = DataLoader(dataset, batch_size=batch_size, sampler=trn_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)


    # Initialize model, criterion, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = len(dataset.chars)
    output_size = len(dataset.chars)

    rnn = CharRNN(input_size, hidden_size, output_size, num_layers).to(device)
    lstm = CharLSTM(input_size, hidden_size, output_size, num_layers).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    rnn_optimizer = optim.Adam(rnn.parameters(), lr=learning_rate)
    lstm_optimizer = optim.Adam(lstm.parameters(), lr=learning_rate)

    rnn_trn_losses, rnn_val_losses = [], []
    lstm_trn_losses, lstm_val_losses = [], []
    
    rnn_best_val_loss, lstm_best_val_loss = float('inf'), float('inf')
    rnn_best, lstm_best = None, None
    
    # Training loop
    for epoch in range(num_epochs):
        rnn_trn_loss = train(rnn, trn_loader, device, criterion, rnn_optimizer)
        rnn_val_loss = validate(rnn, val_loader, device, criterion)
        lstm_trn_loss = train(lstm, trn_loader, device, criterion, lstm_optimizer)
        lstm_val_loss = validate(lstm, val_loader, device, criterion)
        
        rnn_trn_losses.append(rnn_trn_loss)
        rnn_val_losses.append(rnn_val_loss)
        lstm_trn_losses.append(lstm_trn_loss)
        lstm_val_losses.append(lstm_val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}, RNN : Trn Loss: {rnn_trn_loss:.4f}, Val Loss: {rnn_val_loss:.4f}, LSTM : Trn Loss: {lstm_trn_loss:.4f}, Val Loss: {lstm_val_loss:.4f}')

        
        if rnn_val_loss < rnn_best_val_loss :
            rnn_val_loss = rnn_best_val_loss
            rnn_best = rnn.state_dict()
        if lstm_val_loss < lstm_best_val_loss :
            lstm_val_loss = lstm_best_val_loss
            lstm_best = rnn.state_dict()

    np.save('../result/rnn_trn_losses.npy', rnn_trn_losses)
    np.save('../result/rnn_val_losses.npy', rnn_val_losses)
    np.save('../result/lstm_trn_losses.npy', lstm_trn_losses)
    np.save('../result/lstm_val_losses.npy', lstm_val_losses)
    
    torch.save(rnn_best, '../result/rnn.pth')
    torch.save(lstm_best,'../result/lstm.pth')
    
if __name__ == '__main__':
    main()