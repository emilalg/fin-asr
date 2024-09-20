import torchaudio
import torch
from alphabet import Alphabet

mel_spectrogram_converter = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=400,
        n_mels=40
)

alphabet = Alphabet()

# from audio_utils
def preprocess_audio(batch):

    max_input_length = 0
    max_label_length = 0

    audio = []
    label = []
    audio_length = []
    label_length = []

    for item in batch:
        audio_array = torch.tensor(item['audio']['array']).float()

        sample_rate = item['audio']['sampling_rate']
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            audio_array = resampler(audio_array)
        
        audio_array = torchaudio.functional.preemphasis(audio_array)

        mel_spectrogram = mel_spectrogram_converter(audio_array)

        # Apply log to the mel spectrogram
        mel_spectrogram = torch.log(mel_spectrogram + 1e-9)

        # Normalize the spectrogram
        mel_spectrogram = (mel_spectrogram - mel_spectrogram.mean()) / mel_spectrogram.std()

        # Transpose the mel spectrogram to correct dimension (time, mels)
        mel_spectrogram = mel_spectrogram.transpose(0, 1)

        # eka bugi löyty (väärä shape (1))
        max_input_length = max(max_input_length, mel_spectrogram.shape[0])

        sentence = torch.tensor(alphabet.text_to_array(str.lower(item['sentence'])))

        max_label_length = max(max_label_length, len(sentence))

        audio.append(mel_spectrogram)
        label.append(sentence)
        audio_length.append(mel_spectrogram.shape[0])
        label_length.append(len(sentence))

    audio_padded = map(lambda x: torch.nn.functional.pad(x, (0, 0, 0, max_input_length - x.size(0))), audio)

    blank_index = len(alphabet.alphabet) - 1  # Index of the blank token
    labels_padded = map(lambda x: torch.nn.functional.pad(x, (0, max_label_length - len(x)), value=blank_index), label)

    return (list(audio_padded), list(labels_padded), audio_length, label_length)
