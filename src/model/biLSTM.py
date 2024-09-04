import torch
import torch.nn as nn
import torch.nn.functional as F

class biLSTM(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=320, num_layers=4, num_classes=40, dropout_rate=0.2):
        super(biLSTM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # Stacked bidirectional LSTM layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, bidirectional=True, batch_first=True)
        
        # Output layer
        self.fc = nn.Linear(hidden_dim * 2, num_classes + 1)  # +1 for blank label in CTC
        
        # Dropout masks
        self.forward_dropout_mask = None
        self.recurrent_dropout_mask = None

        self.init_forget_gate_bias()

    def forward(self, x, apply_dropout=True):
        batch_size, seq_len, _ = x.size()
        
        if apply_dropout:
            # Forward dropout (sequence-level)
            if self.forward_dropout_mask is None or self.forward_dropout_mask.size(0) != batch_size:
                self.forward_dropout_mask = torch.bernoulli(torch.ones(batch_size, self.input_dim) * (1 - self.dropout_rate)).to(x.device)
            x = x * self.forward_dropout_mask.unsqueeze(1)
            
            # Recurrent dropout without memory loss (sequence-level)
            if self.recurrent_dropout_mask is None or self.recurrent_dropout_mask.size(0) != batch_size:
                self.recurrent_dropout_mask = torch.bernoulli(torch.ones(batch_size, self.hidden_dim * 2) * (1 - self.dropout_rate)).to(x.device)
        
        # LSTM layers
        lstm_out, _ = self.lstm(x)
        
        if apply_dropout:
            # Apply recurrent dropout without memory loss
            lstm_out = lstm_out * self.recurrent_dropout_mask.unsqueeze(1)
        
        # Output layer
        output = self.fc(lstm_out)
        
        # Log softmax for CTC loss
        log_probs = F.log_softmax(output, dim=2)
        
        return log_probs

    def init_forget_gate_bias(self):
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)