import torch
import torch.nn as nn
import torch.nn.functional as F

# https://github.com/itberrios/3D/blob/main/point_net/point_net.py
# PCN: Point Completion Network

class PointNet(nn.Module):
    def __init__(self, num_global_feats=1024):
        super().__init__()
        self.num_global_feats = num_global_feats

        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )

        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.num_global_feats, 1),
            nn.BatchNorm1d(self.num_global_feats),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, return_g=False):
        # x: [B, N, 3] -> [B, 3, N]
        x = x.transpose(1, 2)

        feat = self.first_conv(x)                 # [B, 256, N]
        g = torch.max(feat, dim=2).values         # [B, 256]

        g_exp = g.unsqueeze(-1).expand(-1, -1, feat.size(2))   # [B, 256, N]
        feat2 = torch.cat([feat, g_exp], dim=1)                # [B, 512, N]

        feat2 = self.second_conv(feat2)          # [B, latent_dim, N]
        v = torch.max(feat2, dim=2).values       # [B, latent_dim]  (1024)

        return (v, g) if return_g else v
