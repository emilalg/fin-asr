import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model.basic import FinnishLSTM  # Import your model
from data.hf import FinnishCommonVoiceDataset
from tqdm import tqdm
from utils import text_to_indices, indices_to_text, params, CHAR_TO_INDEX
import os

import logging


def train(model, train_loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
        spectrograms = batch['mel_spectrogram'].to(device)
        labels = [text_to_indices(label) for label in batch['sentence']]
        labels = torch.nn.utils.rnn.pad_sequence([torch.LongTensor(l) for l in labels], batch_first=True).to(device)
        
        optimizer.zero_grad()
        
        output = model(spectrograms)
        
        input_lengths = torch.full(size=(spectrograms.size(0),), fill_value=spectrograms.size(1), dtype=torch.long)
        target_lengths = torch.sum(labels != 0, dim=1)
        
        loss = criterion(output.transpose(0, 1), labels, input_lengths, target_lengths)
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
    # Load datasets
    train_dataset = FinnishCommonVoiceDataset(split="train", data_dir=args.data_dir, cache_spectrograms=True)
    val_dataset = FinnishCommonVoiceDataset(split="validation", data_dir=args.data_dir, cache_spectrograms=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

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
    
    


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()