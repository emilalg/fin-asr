import logging
import math
import os
import gc
import torch
import torch.nn as nn
import torch.utils
import random
from torchinfo import summary
from tqdm import tqdm

from model.model import LSTMCTC

import utils
from alphabet import Alphabet
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F
from audio_utils import preprocess_audio
from data.huggingface import HuggingFaceDataset
from data.kielipankki import KielipankkiDataset
from torch.utils.data import ConcatDataset
from pywer import wer, cer

def train(model, train_loader, device, optimizer, loss_fn, epoch):
    
    # set training mode
    model.train()
    total_loss = 0
    skipped_batches = 0

    # run batch and accumulate loss (train)
    for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} - Training")):
        try:
            audio_padded, labels_padded, audio_lengths, label_lengths = batch
            
            audio = torch.stack(audio_padded, dim=0).to(device)
            # Shape: (batch_size, max_label_length)
            labels = torch.stack(labels_padded, dim=0).to(device)
            input_lengths = torch.tensor(audio_lengths).to(device)
            target_lengths = torch.tensor(label_lengths).to(device)
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(audio)
            
            # Shape: (max_input_length, batch_size, num_classes)
            log_probs = F.log_softmax(outputs, dim=2).permute(1, 0, 2)

            # clamp log probs to potentially avoid nans and infs
            # probabaly leads to a slight disruption in the gradients
            log_probs = log_probs.clamp(min=torch.finfo(log_probs.dtype).min, max=torch.finfo(log_probs.dtype).max)

            # skip batch if any still occur
            if torch.any(torch.isnan(log_probs)) or torch.any(torch.isinf(log_probs)):
                logging.error(f"NaN or infinity detected in log_probs for batch {batch_idx}")
                continue  # Skip this batch and move to the next one

            # Compute loss
            loss = loss_fn(log_probs, labels, input_lengths, target_lengths)
            
            # check loss scalar for nan or inf just in case
            loss_item = loss.item()
            if math.isnan(loss_item) or math.isinf(loss_item):
                logging.warning(f'NaN or infinity detected in loss scalar for batch {batch_idx}')
                continue

            loss.backward()
            
            # clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)  
            optimizer.step()
            total_loss += loss_item

        except torch.cuda.OutOfMemoryError as e:
            logging.warning(f"Ran out of memory in batch {batch_idx}. "
                               f"Audio tensor shape: ({audio.shape}). "
                               f"Skipping batch")
            skipped_batches += 1
            continue

        except RuntimeError as e:
            if "out of memory" in str(e):
                # Log the shape of the audio tensor
                logging.warning(f"Ran out of memory in batch {batch_idx}. "
                               f"Audio tensor shape: ({audio.shape}). "
                               f"Skipping batch")
            else:
                logging.error(f"An unexpected runtime error occurred in batch {batch_idx}: {str(e)}")
            skipped_batches += 1
            continue

        except Exception as e:
            logging.error(f"An unexpected error occurred in batch {batch_idx}: {str(e)}")
            skipped_batches += 1
            continue

        finally:
            # Clean up to help avoid OOM issues
            # Slightly increased runtime, but stabilizes vram usage
            del audio, labels, input_lengths, target_lengths
            if 'outputs' in locals():
                del outputs
            if 'log_probs' in locals():
                del log_probs
            if 'loss' in locals():
                del loss
            torch.cuda.empty_cache()
            gc.collect()

    train_loss = total_loss / (len(train_loader)-skipped_batches)
    return train_loss




def validate(model, val_loader, device, epoch, loss_fn, alphabet, scheduler, ts):
    model.eval()

    total_val_loss = 0
    total_wer = 0
    total_cer = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} - Validating")):
            audio_padded, labels_padded, audio_lengths, label_lengths = batch
   
            audio = torch.stack(audio_padded, dim=0).to(device)
            # Shape: (batch_size, max_label_length)
            labels = torch.stack(labels_padded, dim=0).to(device)
            input_lengths = torch.tensor(audio_lengths).to(device)
            target_lengths = torch.tensor(label_lengths).to(device)

            # compute loss
            output = model(audio)
            log_probs = F.log_softmax(output, dim=2).permute(1, 0, 2)
            loss = loss_fn(log_probs, labels, input_lengths, target_lengths)
            total_val_loss += loss.item()

            # Calculate log_probs for decoding (without permute)
            log_probs_decode = F.log_softmax(output, dim=-1)
            # greedy docoding
            decoded_text = alphabet.decode(log_probs_decode, remove_blanks=True)

            #compute wer & cer
            ref = []
            for label, length in zip(labels, label_lengths):
                clipped_label = label[:length]
                ref.append(alphabet.array_to_text(clipped_label.cpu().numpy()))

            logging.debug(f"""
            ref0 {ref[0]}
            dec0 {decoded_text[0]}
            logprobs decoder shape {log_probs_decode.shape}
            log_probs shape {log_probs.shape}
            loss {loss.item()}
            """)

            # log samples (select different batch for every epoch)
            if batch_idx == epoch:
                rs = random.randint(0, len(decoded_text)-1)
                ts.add_sample(ref[rs], decoded_text[rs], epoch)
            elif epoch > len(val_loader):
                # this block is not getting hit ever :D
                if batch_idx == random.randint(0, len(val_loader)-1):
                    rs = random.randint(0, len(decoded_text)-1)
                    ts.add_sample(ref[rs], decoded_text[rs], epoch)

            
            total_wer += wer(ref, decoded_text)
            total_cer += cer(ref, decoded_text)

            #perhaps helps avoid oom issue
            del audio, labels, input_lengths, target_lengths
            del output, log_probs, log_probs_decode
            del loss
            # maybe helps
            torch.cuda.empty_cache()
            gc.collect()

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

    #for reproducability
    torch.manual_seed(42)

    #init utils
    alphabet = Alphabet()
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

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=preprocess_audio)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=preprocess_audio)

    model = LSTMCTC(
        input_size=40,  # mels
        hidden_size=320,
        num_layers=4,
        num_classes=len(alphabet.alphabet),
        dropout_rate=0.2
    ).to(device)

    #2048 placeholder for max sequence length, computing it would be too 
    #expensive due to the size of the dataset
    summary(model, input_size=(args.batch_size, 2048, 40))

    # initialize optimizer, loss_fn, and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    #calculate index of blank token to pass to CTCLOSS
    blank_index = len(alphabet.alphabet) - 1
    loss_fn = nn.CTCLoss(blank=blank_index, zero_infinity=True).to(device)

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