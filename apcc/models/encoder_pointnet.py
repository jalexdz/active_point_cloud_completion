import torch
import torch.nn as nn
import torch.nn.functional as F

# https://github.com/itberrios/3D/blob/main/point_net/point_net.py

class PointNet(nn.Module):
    def __init__(self, num_global_feats=1024):
        '''Parameters:
            num_points: number of input points in point cloud
            num_global_feats: dimension of global feature vector
            local_feat: whether to concatenate local and global features
        '''
        super(PointNet, self).__init__()

        self.num_global_feats = num_global_feats

        # Shared MLP 1
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1)

        # Shared MLP 2
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv5 = nn.Conv1d(128, self.num_global_feats, kernel_size=1)

        # Batch norms
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(self.num_global_feats)

    def forward(self, x):
        """
        x: [B, N, 3] 
        """
        x = x.permute(0, 2, 1) # [B, 3, N]

        x = self.bn1(F.relu(self.conv1(x)))   # [B, 64, N]
        x = self.bn2(F.relu(self.conv2(x)))   # [B, 64, N]

        # Second shared MLP
        x = self.bn3(F.relu(self.conv3(x)))   # [B, 64, N]
        x = self.bn4(F.relu(self.conv4(x)))   # [B, 128, N]
        x = self.bn5(F.relu(self.conv5(x)))   # [B, 1024, N] 

        global_features = torch.max(x, dim=2)[0]
 
        return global_features