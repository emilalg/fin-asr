#https://arxiv.org/pdf/1707.00722v2

import torch
import torch.nn as nn

class FinnishLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate=0.2):
        super(FinnishLSTM, self).__init__()
        

        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.lstm3 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        self.lstm4 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.dropout4 = nn.Dropout(dropout_rate)
        
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        
        x, _ = self.lstm3(x)
        x = self.dropout3(x)
        
        x, _ = self.lstm4(x)
        x = self.dropout4(x)
        
        # Apply the fully connected layer to each time step
        x = self.fc(x)
        
        # Apply log_softmax along the character dimension
        log_probs = torch.log_softmax(x, dim=2)
        
        return log_probs
    