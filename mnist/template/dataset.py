# import some packages you need here
import os
import torch
import PIL
from torch.utils.data import Dataset
from torchvision import transforms

class MNIST(Dataset):
    """ MNIST dataset

        To write custom datasets, refer to
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        data_dir: directory path containing images

    Note:
        1) Each image should be preprocessed as follows:
            - First, all values should be in a range of [0,1]
            - Substract mean of 0.1307, and divide by std 0.3081
            - These preprocessing can be implemented using torchvision.transforms
        2) Labels can be obtained from filenames: {number}_{label}.png
    """

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_list = [os.path.join(data_dir, data) for data in os.listdir(data_dir)] # data list
        self.transform =  transforms.Compose([transforms.ToTensor(),  # 이미지를 텐서로 변환, 자동으로 0~1로 Normalize
                                              transforms.Normalize((0.1307,), (0.3081,))])  # 평균값 빼기 및 표준 편차로 나누기
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        target = self.data_list[idx]
        label = int(target.split('_')[1].split('.')[0])
        
        image = PIL.Image.open(target).convert('L') 
        image = self.transform(image)
        
        return image, label

class CustomMNIST(Dataset):
    def __init__(self, data_dir,train=True):
        self.data_dir = data_dir
        self.data_list = [os.path.join(data_dir, data) for data in os.listdir(data_dir)] # data list
        if train:
            self.transform =  transforms.Compose([transforms.ToTensor(),  # 이미지를 텐서로 변환, 자동으로 0~1로 Normalize
                                                  transforms.RandomRotation(degrees=(-10, 10)), # # Data Augmentation - 랜덤 회전
                                                  transforms.Normalize((0.1307,), (0.3081,))])  # 평균값 빼기 및 표준 편차로 나누기
        else:
            self.transform =  transforms.Compose([transforms.ToTensor(),  # 이미지를 텐서로 변환, 자동으로 0~1로 Normalize
                                              transforms.Normalize((0.1307,), (0.3081,))])  # 평균값 빼기 및 표준 편차로 나누기
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        target = self.data_list[idx]
        label = int(target.split('_')[1].split('.')[0])
        
        image = PIL.Image.open(target).convert('L') 
        image = self.transform(image)
        
        return image, label

if __name__ == '__main__': 
    dataset = MNIST('../data/train')
    for images in dataset:
        print(images[0].size()) # 이미지 크기 확인
        break

