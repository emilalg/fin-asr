import argparse
from dataclasses import dataclass
from pathlib import Path

@dataclass
class params():
    
    def __init__(self):
        parser = argparse.ArgumentParser(description="Train a speech recognition model on Finnish Common Voice dataset")
        parser.add_argument('--data_dir', type=str, default='data/hf', help='Directory for dataset cache')
        parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory for saving checkpoints')
        parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
        parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
        parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
        parser.add_argument('--num_workers', type=int, default=4, help='Number of worker processes for data loading')
        parser.add_argument('--model_name', type=str, default="test", help='Name of the model, used for output destination')
        self.args = parser.parse_args()

        self.args.output = f'output/{self.args.model_name}/'
        Path(self.args.output).mkdir(parents=True, exist_ok=True)
        



class alphabet():

    def __init__(self, alphabet = "abcdefghijklmnopqrstuvwxyzåäö"):
        self.CHAR_TO_INDEX = {char: i+1 for i, char in enumerate(alphabet)}
        self.INDEX_TO_CHAR = {i+1: char for i, char in enumerate(alphabet)}
        self.CHAR_TO_INDEX['<blank>'] = 0
        self.INDEX_TO_CHAR[0] = '<blank>'
        self.length = len(self.CHAR_TO_INDEX)

    def text_to_indices(self, text):
        return [self.CHAR_TO_INDEX.get(char.lower(), self.CHAR_TO_INDEX['<blank>']) for char in text]
    
    def indices_to_text(self, indices):
        return ''.join([self.INDEX_TO_CHAR[i] for i in indices if i != 0])




class metrics():

    def __init__(self) -> None:
        self.train_loss = []
        self.val_loss = []

    def append_losses(self, loss_t, loss_v):
        self.train_loss.append(loss_t)
        self.val_loss.append(loss_v)