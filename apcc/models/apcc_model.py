import torch
import torch.nn as nn
from .encoder_pointnet import PointNetPP
from .belief_update import BeliefUpdate
from .decoder import APCCDecoder

class APCCModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def init_belief(self, batch_size, device):
        pass

    def update_belief(self, m_t, pc_t):
        pass

    def query_occupancy(self, m_t, query_points):
        pass