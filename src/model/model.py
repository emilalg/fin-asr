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