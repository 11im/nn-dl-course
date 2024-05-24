import os
import dataset
from model import LeNet5, CustomMLP, CustomLeNet5
import torch
import torch.nn.functional
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

def train(model, trn_loader, device, criterion, optimizer):
    """ Train function
    
    Args:
        model: network
        trn_loader: torch.utils.data.DataLoader instance for training
        device: device for computing, cpu or gpu
        criterion: cost function
        optimizer: optimization method, refer to torch.optim
    
    Returns:
        trn_loss: average loss value
        acc: accuracy
    """
    
    trn_loss = 0
    correct = 0
    total_data = 0
    
    model.train()
    
    # data loader 반복문
    for data, target in trn_loader:

        data= data.to(device)
        target = target.to(device)        
        
        # Gradient 초기화
        optimizer.zero_grad()
        
        # 예측
        output = model(data)\
        
        # Loss
        loss = criterion(output, target)
        
        # backpropagation
        loss.backward()
        
        # 파라미터 업데이트
        optimizer.step()
        
        # 손실 값 누적
        trn_loss += loss.item()
        
        # argmax를 사용하여 인덱스 추출 후 비교
        predicted = torch.argmax(output, dim=1)
        correct += (predicted == target).sum().item()
        
        # 총 데이터 수
        total_data += data.size(0)
    
    # Average Loss
    trn_loss /= total_data
    
    # Accuracy
    acc = correct / total_data
    
    return trn_loss, acc

def test(model, tst_loader, device, criterion):
    """ Test function

    Args:
        model: network
        tst_loader: torch.utils.data.DataLoader instance for testing
        device: device for computing, cpu or gpu
        criterion: cost function

    Returns:
        tst_loss: average loss value
        acc: accuracy
    """
    
    tst_loss = 0
    correct = 0
    total_data = 0
      
    model.eval()
   
    # data loader 반복문
    with torch.no_grad():
        for data, target in tst_loader:
            data= data.to(device)
            target = target.to(device)
            
            # 예측
            output = model(data)
            
            # Loss
            loss = criterion(output, target)
            tst_loss += loss.item()
            
            # argmax를 사용하여 인덱스 추출 후 비교
            predicted = torch.argmax(output, dim=1)
            correct += (predicted == target).sum().item()
            
            # 총 데이터 수
            total_data += data.size(0)

    # Average Loss
    tst_loss /= total_data
    
    # Accuracy
    acc = correct / total_data
    
    return tst_loss, acc

def main():
    """ Main function

        Here, you should instantiate
        1) Dataset objects for training and test datasets
        2) DataLoaders for training and testing
        3) model
        4) optimizer: SGD with initial learning rate 0.01 and momentum 0.9
        5) cost function: use torch.nn.CrossEntropyLoss

    """
    
    # tar.sh 스크립트를 통해 압축 해제
    # os.system("../unzip.sh")
    
    batch_size = 64
    learning_rate = 0.01
    momentum = 0.9
    epochs = 10 
    train_loss_history, test_loss_history, train_accuracy_history, test_accuracy_history = [],[],[],[]

    device = torch.device("cuda:0")

    # train_set = dataset.MNIST('../data/train')
    # test_set = dataset.MNIST('../data/test')
    train_set = dataset.CustomMNIST('../data/train')
    test_set = dataset.CustomMNIST('../data/test')

    train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(test_set, batch_size = batch_size, shuffle = False)
    
    # model = CustomMLP().to(device)
    # model = LeNet5().to(device)
    model = CustomLeNet5().to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum = momentum)
    criterion = torch.nn.CrossEntropyLoss()
    
    for e in tqdm(range(epochs)):
        train_loss, train_acc = train(model = model, trn_loader = train_loader, device = device, criterion = criterion, optimizer = optimizer)
        test_loss, test_acc = test(model = model, tst_loader = test_loader, device = device, criterion = criterion)
        print(f'-----------------------------------------------{e+1}--------------------------------------------')
        print(f'Train Loss : {train_loss:.3f}, Train Accuracy : {train_acc:.5f}, Test Loss : {test_loss:.3f}, Test Accuracy : {test_acc:.5f}')

        train_loss_history.append(train_loss)
        train_accuracy_history.append(train_acc)
        test_loss_history.append(test_loss)
        test_accuracy_history.append(test_acc)
    
    # np.save('../result/CustomMLP/train_loss.npy',train_loss_history)
    # np.save('../result/CustomMLP/test_loss.npy',test_loss_history)
    # np.save('../result/CustomMLP/train_accuracy.npy',train_accuracy_history)
    # np.save('../result/CustomMLP/test_accuracy.npy',test_accuracy_history)

    # np.save('../result/LeNet/train_loss.npy',train_loss_history)
    # np.save('../result/LeNet/test_loss.npy',test_loss_history)
    # np.save('../result/LeNet/train_accuracy.npy',train_accuracy_history)
    # np.save('../result/LeNet/test_accuracy.npy',test_accuracy_history)
    
    np.save('../result/CustomLeNet/train_loss.npy',train_loss_history)
    np.save('../result/CustomLeNet/test_loss.npy',test_loss_history)
    np.save('../result/CustomLeNet/train_accuracy.npy',train_accuracy_history)
    np.save('../result/CustomLeNet/test_accuracy.npy',test_accuracy_history)

if __name__ == '__main__':
    main()
