import torch
import torch.nn as nn
import torch.nn.functional as F

# ------ TO DO ------
class cls_model(nn.Module):
    def __init__(self, num_classes=3):
        super(cls_model, self).__init__()

        self.num_classes = num_classes

        self.mlp1 = nn.Sequential(
            # This kernel and stride size to ensure that the conv layer
            # doesn't take info from other neighbouring point coordinates.
            nn.Conv1d(3, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        self.mlp2 = nn.Sequential(
            # This kernel and stride size to ensure that the conv layer
            # doesn't take info from other neighbouring point coordinates.
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, kernel_size=1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )

        self.mlp3 =  nn.Sequential(
            # This kernel and stride size to ensure that the conv layer
            # doesn't take info from other neighbouring point coordinates.
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, self.num_classes)
        )



    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, num_classes)
        '''

        x = torch.permute(points, (0, 2, 1))
        out = self.mlp1(x)
        out = self.mlp2(out)

        # Global pooling across last dimension
        out = torch.amax(out, dim=-1)
        out = self.mlp3(out)

        return out

# ------ TO DO ------
class seg_model(nn.Module):
    def __init__(self, num_seg_classes = 6):
        super(seg_model, self).__init__()

        self.num_classes = num_seg_classes

        self.mlp1 = nn.Sequential(
            # This kernel and stride size to ensure that the conv layer
            # doesn't take info from other neighbouring point coordinates.
            nn.Conv1d(3, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        self.mlp2 = nn.Sequential(
            # This kernel and stride size to ensure that the conv layer
            # doesn't take info from other neighbouring point coordinates.
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, kernel_size=1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )

        self.mlp3 = nn.Sequential(
            # This kernel and stride size to ensure that the conv layer
            # doesn't take info from other neighbouring point coordinates.
            nn.Conv1d(1088, 512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, self.num_classes, kernel_size=1),
        )

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, N, num_seg_classes)
        '''

        x = torch.permute(points, (0, 2, 1))
        local_feats = self.mlp1(x)

        # Global pooling across last dimension
        global_feats = torch.amax(self.mlp2(local_feats), dim=-1, keepdims=True)
        global_feats = global_feats.repeat(1, 1, x.size(-1))

        out = torch.cat((local_feats, global_feats), dim=1)

        out = self.mlp3(out)

        return torch.permute(out, (0, 2, 1))


