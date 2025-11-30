import torch
import torch.nn as nn

from .encoder_pointnet import PointNet
from .belief_gru import BeliefGRU
from .decoder import APCCDecoder

class APCCModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        enc_feat_dim = cfg.enc_feat_dim
        gru_hidden_dim = cfg.gru_hidden_dim
        num_layers = cfg.gru_layers

        # PointNet encoder
        self.encoder = PointNet(
            num_global_feats=enc_feat_dim
        )

        # GRU
        self.gru = BeliefGRU(
            input_size=enc_feat_dim,
            hidden_size=gru_hidden_dim,
            num_layers=num_layers,
            dropout=cfg.gru_dropout
        )

        # Decoder
        self.decoder = APCCDecoder(
            hidden_dim=gru_hidden_dim,
            query_dim=3,
            mlp_hidden_dim=cfg.dec_mlp_hidden_dim,
            num_layers=cfg.dec_num_layers
        )

    def forward(self, pointcloud_t, query_xyz, h_prev=None):
        # Encode and get feature vector
        e_t = self.encoder(pointcloud_t)  # (B, enc_feat_dim)

        # Update GRU
        h_t, hn = self.gru(e_t, h_prev)  # (B, H), (num_layers, B, H)

        # Decode to get occupancy predictions
        occ_logits = self.decoder(h_t, query_xyz)  # (B, num_query, 1)

        return occ_logits, hn

