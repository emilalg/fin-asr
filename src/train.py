import logging
import torch
import torch.nn as nn
import torchaudio
from tqdm import tqdm
from model.basic import FinnishLSTM
import utils
import datasets
from torch.utils.data import DataLoader
from torch import optim

# collate_fn for the dataloader to process audio into the correct format
def preprocess_audio(batch):

    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []

    alphabet = utils.alphabet()

    # initialize mel converter
    mel_spectrogram_converter = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=400,
        n_mels=40
    )

    max_input_length = 0

    # iterate over batch
    for item in batch:

        # read in waveform and convert to float
        waveform = torch.tensor(item['audio']['array']).float()

        # resample just in case
        sample_rate = item['audio']['sampling_rate']
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)

        # preemphasis for potentially better feature extraction
        waveform = torchaudio.functional.preemphasis(waveform)

        # convert to mel spectrogram
        mel_spectrogram = mel_spectrogram_converter(waveform)

        # Apply log to the mel spectrogram
        mel_spectrogram = torch.log(mel_spectrogram + 1e-9)

        # Normalize the spectrogram
        mel_spectrogram = (mel_spectrogram - mel_spectrogram.mean()) / mel_spectrogram.std()

        # Transpose the mel spectrogram
        mel_spectrogram = mel_spectrogram.transpose(0, 1)

        max_input_length = max(max_input_length, mel_spectrogram.shape[1])

        # read in label
        label = torch.tensor(alphabet.text_to_indices(item['sentence']))

        logging.info(f'Shape of mel {mel_spectrogram.shape} size {mel_spectrogram.size}')

        # append to list
        spectrograms.append(mel_spectrogram)
        labels.append(label)
        input_lengths.append(mel_spectrogram.shape[1])
        label_lengths.append(len(label))
    
    # pad spectrograms
    padded_spectrograms = []
    for spec in spectrograms:
        pad_amount = max_input_length - spec.shape[0]
        padded_spec = torch.nn.functional.pad(spec, (0, 0, 0, pad_amount))  # Pad in time dimension
        padded_spectrograms.append(padded_spec)

    # Padding and stacking
    spectrograms = torch.stack(padded_spectrograms)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

    logging.debug(f'Shapes of tensors.\nSpectrograms: {[s.shape for s in spectrograms]}\nLabels: {[l.shape for l in labels]}.')
    return {
        "spectrograms": spectrograms,
        "labels": labels,
        "input_lengths": torch.tensor(input_lengths),
        "label_lengths": torch.tensor(label_lengths)
    }


# Data and training loop
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    args = utils.params().args
    alphabet = utils.alphabet()
    metrics = utils.metrics()

    # needs hf token added through the huggingface-cli
    train_dataset = datasets.load_dataset(
        "mozilla-foundation/common_voice_17_0",  
        "fi",
        split='train',
        cache_dir=args.data_dir,
        token=True,
        trust_remote_code=True
    )

    val_dataset = datasets.load_dataset(
        "mozilla-foundation/common_voice_17_0",  
        "fi",
        split='validation',
        cache_dir=args.data_dir,
        token=True,
        trust_remote_code=True
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=preprocess_audio)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=preprocess_audio)

    model = FinnishLSTM(
        input_size=40,  # mels
        hidden_size=256,
        num_layers=4,
        num_classes=alphabet.length,
        dropout_rate=0.2
    ).to(device)

    # initialize optimizer, loss_fn, and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

    # training + evaluation loop
    best_val_loss = float('inf')
    for epoch in range(args.epochs):

        # set training mode
        model.train()
        total_loss = 0

        # run batch and accumulate loss (train)
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} - Training"):
            spectrograms = batch['spectrograms'].to(device)
            labels = batch['labels'].to(device)
            input_lengths = batch['input_lengths'].to(device)
            label_lengths = batch['label_lengths'].to(device)

            # :)
            optimizer.zero_grad()

            # forward pass
            output = model(spectrograms)
            output = output.transpose(0, 1).log_softmax(2)  

            # compute loss and run backwards pass
            loss = loss_fn(output, labels, input_lengths, label_lengths)
            loss.backward()
            
            # clip gradients to prevent a possible gradient explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)  

            optimizer.step()
            total_loss += loss.item()
        
        train_loss = total_loss / len(train_loader)

        # validation loop
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} - Validation"):
                spectrograms = batch['spectrograms'].to(device)
                labels = batch['labels'].to(device)
                input_lengths = batch['input_lengths'].to(device)
                label_lengths = batch['label_lengths'].to(device)

                output = model(spectrograms)
                loss = loss_fn(output.transpose(0, 1), labels, input_lengths, label_lengths)
                total_val_loss += loss.item()

        val_loss = total_val_loss / len(val_loader)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Log metrics
        metrics.append_losses(train_loss, val_loss)
        logging.info(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{args.output}/best_model.pth")
            logging.info(f"New best model saved with validation loss: {val_loss:.4f}")

    




if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()