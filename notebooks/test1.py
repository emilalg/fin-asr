# %%
import librosa
import datasets
ds = datasets.load_from_disk('data/hf')

# %%
batch = []
for i in range(0, 8):
    batch.append(ds['train'][i])

print(batch)

# %%
class ASRAlphabet:
    def __init__(self, alphabet_string):
        self.alphabet = list(alphabet_string)
        self.alphabet.append('<blank>')  # Adding blank token
        self.char_to_index = {char: index for index, char in enumerate(self.alphabet)}
        self.index_to_char = {index: char for index, char in enumerate(self.alphabet)}

    def text_to_array(self, text):
        return [self.char_to_index[char] for char in str.lower(text) if char in self.char_to_index]

    def array_to_text(self, array):
        return ''.join(self.index_to_char[index] for index in array if index in self.index_to_char)
    
alphabet = ASRAlphabet(alphabet_string='abcdefghijklmnopqrstuvwxyzåäö ')

# %%
# test alphabet

sample_s = batch[0]['sentence']

aout1 = alphabet.text_to_array(sample_s)
aout2 = alphabet.array_to_text(aout1)

print(sample_s, aout1, aout2)

# %%
import torchaudio
import torch

mel_spectrogram_converter = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=400,
        n_mels=40
)

# from audio_utils
def preprocess_audio(batch):

    max_input_length = 0
    max_label_length = 0

    audio = []
    label = []
    audio_length = []
    label_length = []

    for item in batch:
        t1 = torch.tensor(item['audio']['array']).float()

        sample_rate = item['audio']['sampling_rate']
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            t1 = resampler(t1)
        
        t1 = torchaudio.functional.preemphasis(t1)

        mel_spectrogram = mel_spectrogram_converter(t1)

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



# %%
audio, label, audio_length, label_length = preprocess_audio(batch)
print(len(audio))

# %%
print(audio[0].shape, audio[1].shape)

# %%
print(len(label[1]), len(label[2]))

# %%
import torch.nn as nn
import torch.nn.functional as F

class LSTMCTC(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers, dropout_rate):
        super(LSTMCTC, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout_rate,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        x, _ = self.lstm(x)
        x = self.dropout(x)  
        return self.fc(x)

# %%
from torch.utils.data import DataLoader, ConcatDataset



train_set = ds['train']
train_set1 = ds['validated']

train_loader = DataLoader(ConcatDataset([train_set, train_set1]), batch_size=16, shuffle=True, num_workers=0, collate_fn=preprocess_audio)
val_loader = DataLoader(ds['validation'], batch_size=16, shuffle=False, num_workers=0, collate_fn=preprocess_audio)

# %%
#init model
device = torch.device('cuda')  

model = LSTMCTC(
        input_size=40,  # mels
        hidden_size=320,
        num_layers=4,
        num_classes=len(alphabet.alphabet),
        dropout_rate=0.2
).to(device)


# %%

prev_loss = 999
model.to(device)
blank_index = len(alphabet.alphabet) - 1

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CTCLoss(blank=blank_index, zero_infinity=True)
for epoch in range(50):
    model.train()
    total_loss = 0
    for batch_idx, (audio_padded, labels_padded, audio_lengths, label_lengths) in enumerate(train_loader):
        # Shape: (batch_size, max_input_length, num_mel_features)
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

        # Compute loss
        loss = loss_fn(log_probs, labels, input_lengths, target_lengths)
        
        # Backward pass and optimize
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)  
        optimizer.step()
        
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/50], Loss: {avg_loss:.4f}')
    if prev_loss > avg_loss:
        torch.save(model.state_dict(), 'test1.pth')
    prev_loss = avg_loss


# %%



