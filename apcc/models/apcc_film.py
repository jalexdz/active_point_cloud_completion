import torch 
import torch.nn as nn
import torch.nn.functional as F

# https://arxiv.org/abs/1709.07871

class APCCFiLM(nn.Module):
    '''A feature wise linear modulation (FiLM) layer for conditioning on belief states.'''

    def __init__(self, hidden_size, token_dim, num_tokens):
        super(APCCFiLM, self).__init__()
        
        self.num_tokens = num_tokens
        self.token_dim = token_dim

        self.base_tokens = nn.Parameter(torch.randn(num_tokens, token_dim))

        # FiLM
        self.gamma = nn.Linear(hidden_size, token_dim)
        self.beta = nn.Linear(hidden_size, token_dim)

    def forward(self, h_t):
        B = h_t.shape[0]

        gamma = self.gamma(h_t).unsqueeze(1)  # (B, 1, token_dim)
        beta = self.beta(h_t).unsqueeze(1)    # (B, 1, token_dim)

        base = self.base_tokens.unsqueeze(0).repeat(B, 1, 1)  # (B, num_tokens, token_dim)

        # Modulation
        R_t = gamma * base + beta  # (B, num_tokens, token_dim)

        return R_t