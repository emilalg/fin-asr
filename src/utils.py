import argparse
from dataclasses import dataclass

@dataclass
class params():
    
    def __init__(self):
        parser = argparse.ArgumentParser(description="Train a speech recognition model on Finnish Common Voice dataset")
        parser.add_argument('--data_dir', type=str, default='data/hf', help='Directory for dataset cache')
        parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory for saving checkpoints')
        parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
        parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
        parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
        parser.add_argument('--num_workers', type=int, default=4, help='Number of worker processes for data loading')
        
        self.args = parser.parse_args()

# Define the Finnish alphabet
FINNISH_ALPHABET = "abcdefghijklmnopqrstuvwxyzåäö"  # Add any additional characters you need
CHAR_TO_INDEX = {char: i+1 for i, char in enumerate(FINNISH_ALPHABET)}
INDEX_TO_CHAR = {i+1: char for i, char in enumerate(FINNISH_ALPHABET)}
CHAR_TO_INDEX['<blank>'] = 0  # Add blank token for CTC
INDEX_TO_CHAR[0] = '<blank>'
        
def text_to_indices(text):
    return [CHAR_TO_INDEX.get(char.lower(), CHAR_TO_INDEX['<blank>']) for char in text]

def indices_to_text(indices):
    return ''.join([INDEX_TO_CHAR[i] for i in indices if i != 0])

