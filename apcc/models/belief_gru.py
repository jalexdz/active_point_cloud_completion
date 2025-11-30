import torch 
import torch.nn as nn
import torch.nn.functional as F

# https://medium.com/@anishnama20/understanding-gated-recurrent-unit-gru-in-deep-learning-2e54923f3e2

class BeliefGRU(nn.Module):
    '''A gated recurrent unit (GRU) for storing and updating belief states.'''
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        super(BeliefGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )

    def forward(self, x, h0=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        output, hn = self.gru(x, h0)
        h_t = output[:, -1, :]
        
        return h_t, hn
    
    def init_hidden(self, batch_size, device):
        device = device or next(self.parameters()).device

        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
