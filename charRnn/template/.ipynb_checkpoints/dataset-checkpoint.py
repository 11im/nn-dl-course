import torch
from torch.utils.data import Dataset

class Shakespeare(Dataset):
    def __init__(self, input_file, seq = 30):
        with open(input_file, 'r') as f:
            txt = f.read()

        self.chars = sorted(list(set(txt)))
        self.ch2idx = {ch: idx for idx, ch in enumerate(self.chars)}
        self.idx2ch = {idx: ch for idx, ch in enumerate(self.chars)}

        self.txt_idx = [self.ch2idx[ch] for ch in txt]

        self.seq = seq
        
        self.data = []
        for i in range(0, len(self.txt_idx) - self.seq):
            input_seq = self.txt_idx[i:i + self.seq]
            target_seq = self.txt_idx[i + 1:i + self.seq + 1]
            self.data.append((input_seq, target_seq))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_seq, target_seq = self.data[idx]
        input_tensor = torch.tensor(input_seq, dtype=torch.long)
        target_tensor = torch.tensor(target_seq, dtype=torch.long)
        return input_tensor, target_tensor


if __name__ == '__main__':
    test = Shakespeare("../data/shakespeare_train.txt")
    print(len(test))