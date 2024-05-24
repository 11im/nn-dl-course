import torch
import torch.nn.functional as F
import numpy as np

from model import CharRNN, CharLSTM
from dataset import Shakespeare

def generate(model, seed_characters, temperature=1.0, length=100, device='cpu'):
    """ Generate characters

    Args:
        model: trained model
        seed_characters: seed characters
        temperature: T
        length: length of the generated sequence
        device: device for computing, cpu or gpu

    Returns:
        samples: generated characters
    """
    model.eval()
    chars = list(seed_characters)
    hidden = model.init_hidden(1).to(device)
    
    # Convert seed characters to indices
    dataset = Shakespeare('../data/shakespeare_train.txt')
    input_indices = [dataset.char_to_idx[ch] for ch in chars]
    input_tensor = torch.tensor(input_indices, dtype=torch.long).unsqueeze(0).to(device)

    for _ in range(length):
        output, hidden = model(input_tensor, hidden)
        output = output[:, -1, :] / temperature
        probabilities = F.softmax(output, dim=-1).data
        char_index = torch.multinomial(probabilities, 1).item()
        chars.append(dataset.idx_to_char[char_index])
        input_tensor = torch.tensor([[char_index]], dtype=torch.long).to(device)

    return ''.join(chars)

def main():
    # Parameters
    input_file = '../data/shakespeare_train.txt'  
    model_path = '../result/rnn.pth'    
    # model_path = '../result/lstm.pth'
    seed_characters_list = ['H', 'T', 'W', 'A', 'M'] 
    temperature = 0.8                
    length = 100                     
    hidden_size = 256               
    num_layers = 2                   
    
    # Load dataset for character mappings
    dataset = Shakespeare(input_file)
    
    # Choose device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the model
    model = CharRNN(len(dataset.chars), hidden_size, len(dataset.chars), num_layers).to(device)
    model.load_state_dict(torch.load(model_path))
    
    # Generate samples
    for seed_char in seed_characters_list:
        print(f"Seed: {seed_char}")
        sample = generate(model, seed_char, temperature, length, device)
        print(sample)
        print()

if __name__ == '__main__':
    main()
