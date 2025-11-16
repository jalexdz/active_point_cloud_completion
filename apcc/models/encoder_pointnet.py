import torch
import torch.nn as nn
import torch.nn.functional as F

class Tnet(nn.Module):
    '''T-Net learns a transformation matrix with specified dimension'''
    def __init__(self, dim, num_points=2500):
        super(Tnet, self).__init__()

        # Dimensions for transform matrix
        self.dim = dim
        self.conv1 = nn.Conv1d(dim, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1)

        self.linear1 = nn.Linear(1024, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, dim ** 2)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.max_pool = nn.MaxPool1d(kernel_size=num_points)

    def forward(self, x):
        bs = x.shape[0]

        # Pass through MLP layers
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.bn3(F.relu(self.conv3(x)))

        # Max pooling
        x = self.max_pool(x).views(bs, -1)

        # Pass through MLP
        x = self.bn4(F.relu(self.linear1(x)))
        x = self.bn5(F.relu(self.linear2(x)))
        x = self.linear3(x)

        # Initialize as identity
        iden = torch.eye(self.dim, requires_grad=True).repeat(bs, 1, 1)

        if x.is_cuda:
            iden = iden.cuda()

        x = x.view(-1, self.dim, self.dim) + iden

class PointNet(nn.Module):
    def __init__(self, num_points=2500, num_global_feats=1024, local_feat=True):
        '''Parameters:
            num_points: number of input points in point cloud
            num_global_feats: dimension of global feature vector
            local_feat: whether to concatenate local and global features
        '''
        super(PointNet, self).__init__()

        self.num_points = num_points
        self.num_global_feats = num_global_feats
        self.local_feat = local_feat

        # Spatial transformer (T-nets)
        self.tnet1 = Tnet(dim=3, num_points=num_points)
        self.tnet2 = Tnet(dim=64, num_points=num_points)

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

        # Max pool
        self.max_pool = nn.MaxPool1d(kernel_size=num_points, return_indices=True)

    def forward(self, x):
        bs = x.shape[0]

        # Input transform
        A_input = self.tnet1(x)

        # First transformation
        x = torch.bmm(x.transpose(2, 1), A_input).transpose(2, 1)

        # Pass through first shared MLP
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))

        # Feature transform
        A_feat = self.tnet2(x)

        # Second transformation
        x = torch.bmm(x.transpose(2, 1), A_feat).transpose(2, 1)

        # Store local features
        local_features = x.clone()

        # Pass through second shared MLP
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.bn4(F.relu(self.conv4(x)))
        x = self.bn5(F.relu(self.conv5(x)))

        # Get global feature vector and critical indices
        global_features, critical_indices = self.max_pool(x)
        global_features = global_features.view(bs, -1)

        if self.local_feat:
            features = torch.cat((local_features,
                                  global_features.unsqueeze(-1).repeat(1, 1, self.num_points)),
                                  dim=1)
            
            return features, critical_indices, A_feat
        else:
            return global_features, critical_indices, A_feat