import argparse
from dataclasses import dataclass
from pathlib import Path
import torch
from torch.utils.tensorboard.writer import SummaryWriter

@dataclass
class params():
    
    def __init__(self):
        parser = argparse.ArgumentParser(description="Train a speech recognition model on Finnish Common Voice dataset")
        parser.add_argument('--data_dir', type=str, default='data', help='Directory for dataset cache')
        parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory for saving checkpoints')
        parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
        parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
        parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate for optimizer')
        parser.add_argument('--num_workers', type=int, default=4, help='Number of worker processes for data loading')
        parser.add_argument('--model_name', type=str, default="fin-asr", help='Name of the model, used for output destination')
        parser.add_argument('--debug', type=bool, default=False, help='Use debug output + smaller dataset')
        parser.add_argument('--token', type=str, default='', help='huggingface token for dataset download')
        parser.add_argument('--predl', type=bool, default=False, help='if hf dataset is pre downloaded')
        self.args = parser.parse_args()

        self.args.output = f'output/{self.args.model_name}/'
        Path(self.args.output).mkdir(parents=True, exist_ok=True)
        




class TensorBoardUtils():

    def __init__(self, log_dir, debug=False):
        self.state = debug
        if not self.state:
            self.writer = SummaryWriter(log_dir)
    
    def log_main(self, loss_t, loss_v, wer, cer, epoch):
        if not self.state:
            self.writer.add_scalar('Loss/train', loss_t, epoch)
            self.writer.add_scalar('Loss/validation', loss_v, epoch)
            self.writer.add_scalar('Metrics/WER', wer, epoch)
            self.writer.add_scalar('Metrics/CER', cer, epoch)
            self.writer.flush()
    
    def add_sample(self, t1, t2, epoch):
        if not self.state:
            self.writer.add_text('Sample', f'Ground truth: {t1}, predicted: {t2}', epoch)