import argparse
import time
import torch
from model import ImplicitModel
from r2n2_custom import R2N2
from  pytorch3d.datasets.r2n2.utils import collate_batched_R2N2
import dataset_location
from pytorch3d.ops import sample_points_from_meshes
import losses


def get_args_parser():
    parser = argparse.ArgumentParser('Singleto3D', add_help=False)
    # Model parameters
    parser.add_argument('--arch', default='resnet18', type=str)
    parser.add_argument('--lr', default=4e-4, type=float)
    parser.add_argument('--max_iter', default=1000, type=int)
    parser.add_argument('--log_freq', default=1000, type=int)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--voxel_res', default=32, type=int)
    parser.add_argument('--save_freq', default=10000, type=int)    
    parser.add_argument('--load_checkpoint', action='store_true')            
    return parser

def preprocess(feed_dict,args):
    images = feed_dict['images'].squeeze(1)
    voxels = feed_dict['voxels'].float()
    ground_truth_3d = voxels

    return images.cuda(), ground_truth_3d.cuda()

def calculate_loss(predictions, ground_truth, args):
    loss = losses.voxel_loss(predictions,ground_truth)
    
    return loss

def train_model(args):
    r2n2_dataset = R2N2("train", dataset_location.SHAPENET_PATH, dataset_location.R2N2_PATH, dataset_location.SPLITS_PATH, return_voxels=True)

    loader = torch.utils.data.DataLoader(
        r2n2_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_batched_R2N2,
        pin_memory=True,
        drop_last=True)
    train_loader = iter(loader)

    model = ImplicitModel(args)
    model.cuda()
    model.train()

    # Getting 32 coordinates between -1 and 1. Moving these to the center of the
    # grid.
    coords = torch.arange(-1, 1, 2. / args.voxel_res) + 1. / args.voxel_res
    x, y, z = torch.meshgrid(coords, coords, coords)
    points = torch.stack((x, y, z), dim=-1).unsqueeze(0).cuda()

    # ============ preparing optimizer ... ============
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)  # to use with ViTs
    start_iter = 0
    start_time = time.time()
    
    print("Starting training !")
    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        if step % len(train_loader) == 0: #restart after one epoch
            train_loader = iter(loader)

        read_start_time = time.time()

        feed_dict = next(train_loader)

        images_gt, ground_truth_3d = preprocess(feed_dict,args)
        read_time = time.time() - read_start_time

        prediction_3d = model(images_gt, points, args)

        loss = calculate_loss(prediction_3d, ground_truth_3d, args)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()

        if (step % args.save_freq) == 0:
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, f'checkpoint_implicit_{step}.pth')

        print("[%4d/%4d]; ttime: %.0f (%.2f, %.2f); loss: %.3f" % (step, args.max_iter, total_time, read_time, iter_time, loss_vis))

    print('Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Singleto3D', parents=[get_args_parser()])
    args = parser.parse_args()
    train_model(args)
