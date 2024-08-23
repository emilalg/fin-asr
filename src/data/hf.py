import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import torchaudio
import os
import numpy as np

class FinnishCommonVoiceDataset(Dataset):
    def __init__(self, split="train", data_dir="data/hf", cache_spectrograms=False):
        self.data_dir = data_dir
        self.split = split
        self.cache_spectrograms = cache_spectrograms
        
        # Ensure the data directory exists
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Load the Finnish subset of the dataset with local caching
        self.dataset = load_dataset(
            "mozilla-foundation/common_voice_7_0",
            "fi",  # Finnish language code
            split=self.split,
            token=True,
            cache_dir=self.data_dir
        )
        self.dataset = self.dataset.map(self.prepare_dataset, desc="preprocess dataset")
        
        # Initialize mel spectrogram converter
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=48000,  # Common Voice sample rate
            n_fft=2048,
            hop_length=512,
            n_mels=80
        )
        
        # Cache for spectrograms
        self.spectrogram_cache = {}
        
        print(f"Finnish Common Voice dataset loaded and cached in: {self.data_dir}")
        print(f"Number of samples in {self.split} set: {len(self.dataset)}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        if self.cache_spectrograms and idx in self.spectrogram_cache:
            mel_spec = self.spectrogram_cache[idx]
        else:
            audio = item['audio']['array']
            sample_rate = item['audio']['sampling_rate']
            
            # Convert audio to tensor
            audio_tensor = torch.from_numpy(audio).float()
            
            # Ensure audio is mono
            if len(audio_tensor.shape) > 1:
                audio_tensor = audio_tensor.mean(dim=0)
            
            # Compute mel spectrogram
            mel_spec = self.mel_spec(audio_tensor)
            
            # Convert to dB scale
            mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)
            
            if self.cache_spectrograms:
                self.spectrogram_cache[idx] = mel_spec

        return {
            'mel_spectrogram': mel_spec,
            'sentence': item['sentence'],
            'client_id': item['client_id'],
            'up_votes': item['up_votes'],
            'down_votes': item['down_votes'],
            'age': item['age'],
            'gender': item['gender'],
            'accent': item['accent'],
            'locale': item['locale']
        }

    @staticmethod
    def prepare_dataset(batch):
        """Function to preprocess the dataset"""
        transcription = batch["sentence"]
        
        if transcription.startswith('"') and transcription.endswith('"'):
            transcription = transcription[1:-1]
        
        if transcription[-1] not in [".", "?", "!"]:
            transcription = transcription + "."
        
        batch["sentence"] = transcription
        
        return batch

    def get_data_info(self):
        """Method to get information about the dataset and its location"""
        return {
            "data_directory": self.data_dir,
            "language": "Finnish",
            "split": self.split,
            "num_samples": len(self.dataset),
            "features": list(self.dataset.features.keys()),
            "caching_spectrograms": self.cache_spectrograms
        }

# Example usage:
# train_dataset = FinnishCommonVoiceDataset(split="train", data_dir="data/hf", cache_spectrograms=True)
# dev_dataset = FinnishCommonVoiceDataset(split="validation", data_dir="data/hf", cache_spectrograms=False)
# test_dataset = FinnishCommonVoiceDataset(split="test", data_dir="data/hf", cache_spectrograms=False)

# To create a DataLoader:
# from torch.utils.data import DataLoader
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

# To get dataset info:
# info = train_dataset.get_data_info()
# print(info)