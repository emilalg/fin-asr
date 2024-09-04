import logging
import numpy as np
import torch
from torch import nn
import torchaudio
import utils

# joo
def process_universal_audio(batch):

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
        label = torch.tensor(alphabet.text_to_indices(alphabet.strip(item['sentence'])))

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

    if spectrograms.dim() != 3 or spectrograms.shape[2] != 40:
        logging.warning(f"Unexpected spectrogram tensor shape. Expected (batch_size, variable_sequence_length, 40), got {spectrograms.shape}")

    return {
        "spectrograms": spectrograms,
        "labels": labels,
        "input_lengths": torch.tensor(input_lengths),
        "label_lengths": torch.tensor(label_lengths)
    }



# process huggingface audio
def preprocess_huggingface_audio(batch):

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




# process kielipankki audio
def preprocess_kielipankki_audio(batch):

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

        # read in waveform
        waveform = item['audio']

        # resample just in case
        if item['sample_rate'] != 16000:
            resampler = torchaudio.transforms.Resample(item['sample_rate'], 16000).float()
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
        label = torch.tensor(alphabet.text_to_indices(item['transcript']))

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