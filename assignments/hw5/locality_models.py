import torch
import torch.nn as nn
import torch.nn.functional as F

from models import *

class AbstractionLayer(nn.Module):

    def __init__(self,
                 args,
                 M,
                 R,
                 num_classes=3,
                 in_feats=3,
                 load_ckpt='./checkpoints/cls/best_model.pt'):
        super(AbstractionLayer, self).__init__()

        self.num_classes = num_classes
        self.device = args.device
        self.task = args.task

        self.M = M
        self.R = R

        self.first_layer = nn.Sequential(
            nn.Conv1d(in_feats, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.pretrained_pointnet = cls_model()

        with open(load_ckpt, 'rb') as f:
            state_dict = torch.load(f, map_location=args.device)
            self.pretrained_pointnet.load_state_dict(state_dict)

        # Removing first conv, BN and relu.
        self.pretrained_pointnet.mlp1 = self.pretrained_pointnet.mlp1[3:]

    def __sampling__(self, points):

        # Points - (B, N, 3 + d)
        # Centroids - (B, M, 3 + d)

        xyz = points[..., :3]

        i = torch.randint(low=0, high=xyz.size(1), size=(1,))

        centroids = [points[:, i]]

        for i in range(self.M - 1):
            dists = torch.sqrt(torch.sum(torch.square(xyz - centroids[i][:, :, :3]),
                                         dim=-1))
            inds = torch.argmax(dists, dim=-1)
            centroids.append(points[torch.arange(points.size(0)), inds].unsqueeze(1))

        centroids = torch.cat(centroids, dim=1).to(self.device)

        # print('centroids', centroids.shape)

        return centroids

    def __grouping__(self, points, centroids):

        # Points - (B, N, 3 + d)
        # groups - (B, M, R, 3 + d)

        xyz = points[..., :3].unsqueeze(1)

        # Both shapes - (B, M, N, 3)
        xyz_r = xyz.repeat(1, self.M, 1, 1)
        centroids_r = centroids.unsqueeze(2).repeat(1, 1, xyz.size(1), 1)[...,
                                                                          :3]

        # Shape (B, M, N)
        dists = torch.sqrt(torch.sum(torch.square(xyz_r - centroids_r), dim=-1))

        # Finding top R for each centroid - (B, N, R)
        # Going with KNN for now
        inds = torch.argsort(dists, dim=-1)[..., : self.R]

        groups = []

        for i in range(inds.size(0)):
            group = []
            for j in range(inds.size(1)):
                group.append(points[i, inds[i, j]].unsqueeze(0))

            groups.append(torch.cat(group).unsqueeze(0))

        # Shape - (B, M, R, 3 + d)
        groups = torch.cat(groups)

        # print('groupings', groups.shape)

        return groups

    def __pointnet__(self, points):

        # Points - (B, M, R, 3 + d)
        # x - (B, 3 + d, R, M)
        x = torch.permute(points, (0, 3, 2, 1))

        outs = []

        for i in range(self.M):
            out = self.first_layer(x[..., i])
            out = self.pretrained_pointnet.mlp1(out)
            out = self.pretrained_pointnet.mlp2(out)

            # Shape - (B, 1024)
            global_feats = torch.amax(out, dim=-1, keepdims=True)

            outs.append(global_feats.unsqueeze(1))

        # (B, M, 1024)
        outs = torch.cat(outs, dim=1).to(self.device)

        return outs

    def forward(self, points):

        # Points - (B, N, 3 + d)
        # outs - (B, M, d)

        self.centroids = self.__sampling__(points)
        self.groups = self.__grouping__(points, self.centroids)

        return self.__pointnet__(self.groups)

class PointNet2_Cls(nn.Module):

    def __init__(self, args, num_classes=3):
        super(PointNet2_Cls, self).__init__()

        self.num_classes = num_classes
        self.device = args.device
        self.task = args.task
        self.num_points = 10000

        self.M1 = self.num_points // 100
        self.K1 = self.num_points // 100

        self.M2 = self.num_points // 1000
        self.K2 = self.num_points // 1000

        self.sa1 = AbstractionLayer(args, self.M1, self.K1, num_classes)
        self.sa2 = AbstractionLayer(args,
                                    self.M2,
                                    self.K2,
                                    num_classes,
                                    in_feats=(3 + 1024))

        # Output from sa2 - (B, M2, 1024)
        # global pooling to 1024
        self.mlp1 = nn.Sequential(
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

        out = self.sa1(points)

        # print(self.sa1.centroids[..., :3].shape, out.shape)

        out = self.sa2(torch.cat([self.sa1.centroids[..., :3], out[..., 0]], dim=-1))

        out = torch.permute(out[..., 0], (0, 2, 1))

        # global feat - (B, 1024)
        global_feat = torch.amax(out, dim=-1)

        out = self.mlp1(global_feat)

        return out