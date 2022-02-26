import argparse
import dataset_location
import pytorch3d
import torch
import torch.nn as nn

from pytorch3d.datasets.r2n2.utils import collate_batched_R2N2
from pytorch3d.utils import ico_sphere
from r2n2_custom import R2N2
from torchvision import models as torchvision_models
from torchvision import transforms
from utils_viz import visualize_point_cloud

class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(self.shape)

class SingleViewtoPointCloud(nn.Module):
    def __init__(self, args):
        super(SingleViewtoPointCloud, self).__init__()
        self.device = "cuda"
        vision_model = torchvision_models.__dict__[args.arch](pretrained=True)
        self.encoder = torch.nn.Sequential(*(list(vision_model.children())[:-1]))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

        self.n_point = args.n_points
        # TODO:
        self.decoder = nn.Sequential(
          nn.Linear(512, 1024),
          nn.ReLU(),
          nn.Linear(1024, self.n_point * 3),
          Reshape((-1, self.n_point, 3))
        )

    def forward(self, images, args):
        images_normalize = self.normalize(images.permute(0,3,1,2))
        encoded_feat = self.encoder(images_normalize).squeeze(-1).squeeze(-1)

        pointclouds_pred = self.decoder(encoded_feat)
        return pointclouds_pred

    def forward_encoder(self, images):
        images_normalize = self.normalize(images.permute(0,3,1,2))
        encoded_feat = self.encoder(images_normalize).squeeze(-1).squeeze(-1)

        return encoded_feat

    def forward_decoder(self, encoded_feat):
        pointclouds_pred = self.decoder(encoded_feat)
        return pointclouds_pred

def get_args_parser():
    parser = argparse.ArgumentParser('Singleto3D', add_help=False)
    parser.add_argument('--arch', default='resnet18', type=str)
    parser.add_argument('--type', default='point', type=str)
    parser.add_argument('--n_points', default=5000, type=int)
    parser.add_argument('--index1', default=0, type=int)
    parser.add_argument('--index2', default=0, type=int)
    parser.add_argument('--load_step', default=10000, type=int)
    return parser

def modify_latent_space(args):

    r2n2_dataset = R2N2("test",
                        dataset_location.SHAPENET_PATH,
                        dataset_location.R2N2_PATH,
                        dataset_location.SPLITS_PATH,
                        return_voxels=True)

    # loader = torch.utils.data.DataLoader(
    #     r2n2_dataset,
    #     batch_size=1,
    #     num_workers=2,
    #     collate_fn=collate_batched_R2N2,
    #     pin_memory=True,
    #     drop_last=True)
    # eval_loader = iter(loader)

    model = SingleViewtoPointCloud(args)
    model.cuda()
    model.eval()

    checkpoint = torch.load(f'checkpoint_{args.type}_{args.load_step}.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    image1 = r2n2_dataset[args.index1]['images'].cuda()
    image2 = r2n2_dataset[args.index2]['images'].cuda()

    encoded1 = model.forward_encoder(image1.unsqueeze(0))
    encoded2 = model.forward_encoder(image2.unsqueeze(0))

    high_encoded_feat1 = torch.ones_like(encoded1) * torch.max(encoded1)
    pred = model.forward_decoder(high_encoded_feat1)

    visualize_point_cloud(pred.cpu().detach(),
                          output_path=f'results/interpret_high_1_{args.type}.gif')

    high_encoded_feat2 = torch.ones_like(encoded2) * torch.max(encoded2)
    pred = model.forward_decoder(high_encoded_feat2)

    visualize_point_cloud(pred.cpu().detach(),
                          output_path=f'results/interpret_high_2_{args.type}.gif')

    for w in torch.arange(0, 1.1, 0.25):

        weighted_encoded_feat = (w * encoded1 + (1 - w) * encoded2)
        pred = model.forward_decoder(weighted_encoded_feat)

        visualize_point_cloud(pred.cpu().detach(),
                              output_path=f'results/interpret_{w}_{args.type}.gif')

        zero_encoded_feat = torch.zeros_like(encoded1)
        pred = model.forward_decoder(zero_encoded_feat)

        visualize_point_cloud(pred.cpu().detach(),
                              output_path=f'results/interpret_decoder_{w}_{args.type}.gif')


        # encoded_feat = encoded_dist
        # encoded_feat[encoded_dist < 0] = 0
        # pred = model.forward_decoder(encoded_feat)

        # visualize_point_cloud(pred.cpu().detach(),
        #                       output_path=f'results/interpret_pos_{w}_{args.type}.gif')

        # encoded_feat = encoded_dist
        # encoded_feat[encoded_dist >= 0] = 0
        # pred = model.forward_decoder(encoded_feat)

        # visualize_point_cloud(pred.cpu().detach(),
        #                       output_path=f'results/interpret_neg_{w}_{args.type}.gif')

        # encoded_feat = torch.abs(encoded_dist)
        # encoded_feat[encoded_feat < torch.mean(encoded_feat)] = 0
        # pred = model.forward_decoder(encoded_feat)

        # visualize_point_cloud(pred.cpu().detach(),
        #                       output_path=f'results/interpret_high_{w}_{args.type}.gif')

        # encoded_feat = torch.abs(encoded_dist)
        # encoded_feat[encoded_feat >= torch.mean(encoded_feat)] = 0
        # pred = model.forward_decoder(encoded_feat)

        # visualize_point_cloud(pred.cpu().detach(),
        #                       output_path=f'results/interpret_low_{w}_{args.type}.gif')

if __name__=='__main__':

    parser = argparse.ArgumentParser('Singleto3D', parents=[get_args_parser()])
    args = parser.parse_args()
    modify_latent_space(args)