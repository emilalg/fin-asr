import argparse
from dataclasses import dataclass
from pathlib import Path
from torch.utils.tensorboard.writer import SummaryWriter
import torch

@dataclass
class params():
    
    def __init__(self):
        parser = argparse.ArgumentParser(description="Train a speech recognition model on Finnish Common Voice dataset")
        parser.add_argument('--data_dir', type=str, default='data/', help='Directory for dataset cache')
        parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory for saving checkpoints')
        parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
        parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
        parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
        parser.add_argument('--num_workers', type=int, default=4, help='Number of worker processes for data loading')
        parser.add_argument('--model_name', type=str, default="test1", help='Name of the model, used for output destination')
        parser.add_argument('--debug', type=bool, default=False, help='Use debug output + smaller dataset')
        self.args = parser.parse_args()

        self.args.output = f'output/{self.args.model_name}/'
        Path(self.args.output).mkdir(parents=True, exist_ok=True)
        



class alphabet():
    def __init__(self, alphabet="abcdefghijklmnopqrstuvwxyzåäö "):
        self.CHAR_TO_INDEX = {char: i + 1 for i, char in enumerate(alphabet)}  # Start from 1
        self.INDEX_TO_CHAR = {i + 1: char for i, char in enumerate(alphabet)}  # Start from 1
        self.CHAR_TO_INDEX['<blank>'] = 0  # Blank token at index 0
        self.INDEX_TO_CHAR[0] = '<blank>'
        self.length = len(self.CHAR_TO_INDEX)
        self.alphabet_set = set(alphabet)
   
    def text_to_indices(self, text):
        return [self.CHAR_TO_INDEX.get(char.lower(), 0) for char in text]
   
    def indices_to_text(self, indices):
        return ''.join([self.INDEX_TO_CHAR[int(i)] for i in indices if int(i) != 0]).strip()
   
    def get_labels(self):
        return "".join([self.INDEX_TO_CHAR[i] for i in range(1, self.length)])  # Exclude blank
   
    def strip(self, text):
        return ''.join(char for char in text if char.lower() in self.alphabet_set)

    def greedy_decode(self, logits):
        if isinstance(logits, torch.Tensor):
            if logits.dim() == 2:
                return self._greedy_decode_single(logits)
            elif logits.dim() == 3:
                return [self._greedy_decode_single(seq_logits) for seq_logits in logits]
        raise ValueError("Input must be a 2D (single sequence) or 3D (batch of sequences) tensor")

    def _greedy_decode_single(self, logits):
        indices = torch.argmax(logits, dim=-1).tolist()
        decoded = []
        previous = None
        for index in indices:
            if index != previous and index != 0:  # 0 is blank
                decoded.append(self.INDEX_TO_CHAR[index])
            previous = index
        return ''.join(decoded).strip()




class TensorBoardUtils():

    def __init__(self, log_dir, debug=False):
        self.state = debug
        self.writer = SummaryWriter(log_dir)
    
    def log_main(self, loss_t, loss_v, wer, cer, epoch):
        if not self.state:
            self.writer.add_scalar('Loss/train', loss_t, epoch)
            self.writer.add_scalar('Loss/validation', loss_v, epoch)
            self.writer.add_scalar('WER', wer, epoch)
            self.writer.add_scalar('CER', cer, epoch)
            self.writer.flush()
    
    def add_sample(self, t1, t2, epoch):
        if not self.state:
            self.writer.add_text('Sample', f'Ground truth: {t1}, predicted: {t2}', epoch)