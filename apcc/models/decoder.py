import torch
import torch.nn as nn
import torch.nn.functional as F

class APCCDecoder(nn.Module):
    '''MLP decoder for occupancy prediction from GRU belief state'''

    def __init__(
            self,
            hidden_dim: int = 256,
            query_dim: int = 3,
            mlp_hidden_dim: int = 128,
            num_layers: int = 3,
    ):
        super().__init__()

        in_dim = hidden_dim + query_dim

        layers = []
        last_dim = in_dim

        for i in range(num_layers - 1):
            layers.append(nn.Linear(last_dim, mlp_hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            last_dim = mlp_hidden_dim

        layers.append(nn.Linear(last_dim, 1))

        self.mlp = nn.Sequential(*layers)

    def forward(self, h_t: torch.Tensor, query_xyz: torch.Tensor) -> torch.Tensor:
        B, Nq, _ = query_xyz.shape
        H = h_t.shape[-1]

        h_t_expanded = h_t.unsqueeze(1).expand(-1, Nq, -1)

        x = torch.cat([query_xyz, h_t_expanded], dim=-1)

        x_flat = x.view(B*Nq, -1)
        logits_flat = self.mlp(x_flat)

        occ_logits = logits_flat.view(B, Nq, 1)
        return occ_logits
    