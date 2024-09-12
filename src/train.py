import logging
import os
import torch
import torch.nn as nn
import torch.utils
import random
from tqdm import tqdm

from model.model import LSTMCTC

import utils
from torch.utils.data import DataLoader
from torch import optim
from audio_utils import process_universal_audio
from data.huggingface import HuggingFaceDataset
from data.kielipankki import KielipankkiDataset
from torch.utils.data import ConcatDataset
from pywer import wer, cer

def train(model, train_loader, device, optimizer, loss_fn, epoch):
    
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
        output = output.transpose(0, 1)

        # compute loss and run backwards pass
        loss = loss_fn(output, labels, input_lengths, label_lengths)
        loss.backward()
        
        # clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)  

        optimizer.step()
        total_loss += loss.item()
        
    train_loss = total_loss / len(train_loader)
    return train_loss




def validate(model, val_loader, device, epoch, loss_fn, alphabet, scheduler, ts):
    model.eval()

    total_val_loss = 0
    total_wer = 0
    total_cer = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} - Validation"):
            spectrograms = batch['spectrograms'].to(device)
            labels = batch['labels'].to(device)
            input_lengths = batch['input_lengths'].to(device)
            label_lengths = batch['label_lengths'].to(device)

            # compute loss
            output = model(spectrograms)
            loss = loss_fn(output.transpose(0, 1), labels, input_lengths, label_lengths)
            total_val_loss += loss.item()

            #decode model output to strings
            decoded_text = alphabet.greedy_decode(output)

            #compute wer & cer
            ref = []
            for label, length in zip(labels, label_lengths):
                clipped_label = label[:length]  # Clip the label using its length
                ref.append(alphabet.indices_to_text(clipped_label))

            # log samples randomly
            if random.randint(0, 4) == 4:
                rs = random.randint(0, len(decoded_text)-1)
                ts.add_sample(ref[rs], decoded_text[rs], epoch)

            logging.debug(f' ref {ref} dec {decoded_text}')
            total_wer += wer(ref, decoded_text)
            total_cer += cer(ref, decoded_text)

    val_loss = total_val_loss / len(val_loader)
    cer_out = total_cer / len(val_loader)
    wer_out = total_wer / len(val_loader)
    
    # Update learning rate
    scheduler.step(val_loss)

    return {
        'val_loss' : val_loss,
        'cer' : cer_out,
        'wer' : wer_out
    }




# Data and training loop
def main(args):
    if not torch.cuda.is_available():
        raise Exception('Gpu not available')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    alphabet = utils.alphabet()
    labels = alphabet.get_labels()
    ts = utils.TensorBoardUtils(f'{args.output}log/', args.debug)

    logging.debug(f"Current working directory: {os.getcwd()}")
    logging.debug(f"Data directory: {args.data_dir}")
    logging.debug(f"Contents of data directory: {os.listdir(args.data_dir)}")

    hf_datasets = HuggingFaceDataset(token=args.token, data_dir=args.data_dir + '/hf', predl=args.predl)
    
    # if debug use small dataset
    if args.debug:
        train_set = hf_datasets.train_dataset
        val_set = hf_datasets.val_dataset
    else:
        kp_dataset_train = KielipankkiDataset(mode='train' , data_dir=args.data_dir + '/kielipankki')
        kp_dataset_val = KielipankkiDataset(mode='validation' , data_dir=args.data_dir + '/kielipankki')
        train_set = ConcatDataset([hf_datasets.train_dataset, kp_dataset_train])
        val_set = ConcatDataset([hf_datasets.val_dataset, kp_dataset_val])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=process_universal_audio)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=process_universal_audio)

    model = LSTMCTC(
        input_dim=40,  # mels
        hidden_dim=320,
        num_layers=4,
        num_classes=alphabet.length,
        dropout_rate=0.2
    ).to(device)

    # initialize optimizer, loss_fn, and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

    # training loop
    best_val_loss = float('inf')
    for epoch in range(args.epochs):

        train_loss = train(model, train_loader, device, optimizer, loss_fn, epoch)
        val_metrics = validate(model, val_loader, device, epoch, loss_fn, alphabet, scheduler, ts)

        val_loss = val_metrics['val_loss']
        cer_out = val_metrics['cer']
        wer_out = val_metrics['wer']

        print_output = f"""
        ---
        Epoch {epoch+1}/{args.epochs}
        Train Loss: {train_loss:.4f}
        Val Loss: {val_loss:.4f}
        CER: {cer_out}
        WER: {wer_out}
        ---
        """
        logging.info(print_output)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{args.output}/best_model.pth")
            logging.info(f"New best model saved with validation loss: {val_loss:.4f}")
        
        # log to tensorboard
        ts.log_main(train_loss, val_loss, wer_out, cer_out, epoch)
        





if __name__ == '__main__':
    args = utils.params().args
    logging.basicConfig(
        level = logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    main(args)