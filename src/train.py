import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchaudio
from model.basic import FinnishLSTM  # Import your model
from tqdm import tqdm
from utils import text_to_indices, indices_to_text, params, CHAR_TO_INDEX
import datasets
import os
import logging


def collate_fn(batch):
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=44100,
        n_fft=1024,
        hop_length=512,
        n_mels=80
    )

    for item in batch:
        waveform = torch.tensor(item['audio']['array'])
        sample_rate = item['audio']['sampling_rate']
        if sample_rate != 44100:
            resampler = torchaudio.transforms.Resample(sample_rate, 44100)
            waveform = resampler(waveform)
        
        spectrogram = mel_spectrogram(waveform)
        spectrograms.append(spectrogram.squeeze(0).transpose(0, 1))

        label = torch.LongTensor(text_to_indices(item['sentence']))
        labels.append(label)
        
        input_lengths.append(spectrogram.shape[2])
        label_lengths.append(len(label))
    
    spectrograms = torch.nn.utils.rnn.pad_sequence(spectrograms, batch_first=True)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)
    
    return {
        'spectrograms': spectrograms,
        'labels': labels,
        'input_lengths': torch.LongTensor(input_lengths),
        'label_lengths': torch.LongTensor(label_lengths)
    }


def train(model, train_loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
        spectrograms = batch['spectrograms'].to(device)
        labels = batch['labels'].to(device)
        input_lengths = batch['input_lengths'].to(device)
        label_lengths = batch['label_lengths'].to(device)

        optimizer.zero_grad()

        output = model(spectrograms)

        loss = criterion(output.transpose(0, 1), labels, input_lengths, label_lengths)
        loss.backward()

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            spectrograms = batch['mel_spectrogram'].to(device)
            labels = [text_to_indices(label) for label in batch['sentence']]
            labels = torch.nn.utils.rnn.pad_sequence([torch.LongTensor(l) for l in labels], batch_first=True).to(device)
            
            output = model(spectrograms)
            
            input_lengths = torch.full(size=(spectrograms.size(0),), fill_value=spectrograms.size(1), dtype=torch.long)
            target_lengths = torch.sum(labels != 0, dim=1)
            
            loss = criterion(output.transpose(0, 1), labels, input_lengths, target_lengths)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    args = params().args

    try:
        # Load datasets
        logging.info("Attempting to load train dataset...")
        train_dataset = datasets.load_dataset(
            "mozilla-foundation/common_voice_17_0",  
            "fi",
            split='train',
            cache_dir=args.data_dir,
            token=True,
            trust_remote_code=True
        )
        
        logging.info("Attempting to load validation dataset...")
        val_dataset = datasets.load_dataset(
            "mozilla-foundation/common_voice_17_0",  
            "fi",
            split='validation',
            cache_dir=args.data_dir,
            token=True,
            trust_remote_code=True
        )

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

        # Initialize model, optimizer, and loss function
        model = FinnishLSTM(
            input_size=80,  # Assuming 80 mel frequency bins
            hidden_size=256,
            num_layers=4,
            num_classes=len(CHAR_TO_INDEX),
            dropout_rate=0.2
        ).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        criterion = nn.CTCLoss(blank=0, zero_infinity=True)

        # Training loop
        for epoch in range(args.epochs):
            train_loss = train(model, train_loader, optimizer, criterion, device, epoch)
            val_loss = validate(model, val_loader, criterion, device)
            
            logging.info(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss
            }
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt'))

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        logging.error("Stack trace:", exc_info=True)
        raise
    
    


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()