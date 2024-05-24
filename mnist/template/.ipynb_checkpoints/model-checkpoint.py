import torch.nn as nn

class LeNet5(nn.Module):
    """ LeNet-5 (LeCun et al., 1998)

        - For a detailed architecture, refer to the lecture note
        - Freely choose activation functions as you want
        - For subsampling, use max pooling with kernel_size = (2,2)
        - Output should be a logit vector
    """

    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5) # 합성곱
        self.act1 = nn.ReLU()  # 활성화 함수
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Subsampling
        
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5) # 합성곱
        self.act2 = nn.ReLU()  # 활성화 함수
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # Subsampling
       
        self.fc1 = nn.Linear(16 * 4 * 4, 120) # Full pool2의 Output = 4*4
        self.act3 = nn.ReLU()  # 활성화 함수
       
        self.fc2 = nn.Linear(120, 84) # Full
        self.act4 = nn.ReLU()  # 활성화 함수
        
        self.fc3 = nn.Linear(84, 10) # Full
        
    def forward(self, img):
        output = self.conv1(img)
        output = self.act1(output)
        output = self.pool1(output)
        
        output = self.conv2(output)
        output = self.act2(output)
        output = self.pool2(output)
        
        output = output.view(-1, 16 * 4 * 4) # Full에 전달하기 전에 크기 맞춰주기
        output = self.fc1(output)
        output = self.act3(output)
        
        output = self.fc2(output)
        output = self.act4(output)
        
        output = self.fc3(output)
        
        return output

class CustomMLP(nn.Module):
    def __init__(self):
        super(CustomMLP, self).__init__()

        self.fc1 = nn.Linear(28 * 28, 54)
        self.act1 = nn.ReLU()
        
        self.fc2 = nn.Linear(54, 28)
        self.act2 = nn.ReLU()

        self.fc3 = nn.Linear(28, 10)

    def forward(self, img):
        output = img.view(img.size(0), -1)  # Flatten the input
        
        output = self.fc1(output)
        output = self.act1(output)
        
        output = self.fc2(output)
        output = self.act2(output)
        
        output = self.fc3(output)
        
        return output

class CustomLeNet5(nn.Module):
    def __init__(self):
        super(CustomLeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5) # 합성곱
        self.act1 = nn.ReLU()  # 활성화 함수
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Subsampling
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5) # 합성곱
        self.act2 = nn.ReLU()  # 활성화 함수
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # Subsampling
       
        self.fc1 = nn.Linear(16 * 4 * 4, 120) # Full pool2의 Output = 4*4
        self.act3 = nn.ReLU()  # 활성화 함수
        self.drop1 = nn.Dropout(p=0.3)

        self.fc2 = nn.Linear(120, 84) # Full
        self.act4 = nn.ReLU()  # 활성화 함수
        self.drop2 = nn.Dropout(p=0.3)

        self.fc3 = nn.Linear(84, 10) # Full
         
    def forward(self, img):
        
        output = self.conv1(img)
        output = self.act1(output)
        output = self.pool1(output)
        
        output = self.conv2(output)
        output = self.act2(output)
        output = self.pool2(output)
        
        output = output.view(-1, 16 * 4 * 4) # Full에 전달하기 전에 크기 맞춰주기
        output = self.fc1(output)
        output = self.act3(output)
        output = self.drop1(output)
        
        output = self.fc2(output)
        output = self.act4(output)
        output = self.drop2(output)
        
        output = self.fc3(output)
        
        return output

def count_parameters(model):
    total_params = 0
    params =[]
    for param in model.parameters():
        params.append(param.numel())
        total_params += param.numel()
    return params,total_params

if __name__ == '__main__' :
    lenet = LeNet5()
    print(lenet)
    print(count_parameters(lenet))
    print("----------------------")
    mlp = CustomMLP()
    print(mlp)
    print(count_parameters(mlp))
