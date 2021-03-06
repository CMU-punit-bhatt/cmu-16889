import argparse
import time
import torch
from model import SingleViewto3D, ImplicitModel
from r2n2_custom import R2N2
from  pytorch3d.datasets.r2n2.utils import collate_batched_R2N2
import dataset_location
import pytorch3d
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.ops import knn_points
import mcubes
import utils_vox
import matplotlib.pyplot as plt

from utils_viz import (
    visualize_voxels_as_mesh_image,
    visualize_point_cloud_image,
    visualize_mesh_image,
    visualize_voxels_as_mesh,
    visualize_point_cloud,
    visualize_mesh
)

def get_args_parser():
    parser = argparse.ArgumentParser('Singleto3D', add_help=False)
    parser.add_argument('--arch', default='resnet18', type=str)
    parser.add_argument('--max_iter', default=10000, type=int)
    parser.add_argument('--vis_freq', default=1000, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--type', default='vox', choices=['vox', 'point', 'mesh', 'implicit'], type=str)
    parser.add_argument('--n_points', default=5000, type=int)
    parser.add_argument('--w_chamfer', default=1.0, type=float)
    parser.add_argument('--w_smooth', default=0.1, type=float)
    parser.add_argument('--load_checkpoint', action='store_true')
    parser.add_argument('--load_step', default=10000, type=int)
    parser.add_argument('--voxel_res', default=32, type=int)
    return parser

def preprocess(feed_dict):
    for k in ['images']:
        feed_dict[k] = feed_dict[k].cuda()

    images = feed_dict['images'].squeeze(1)
    mesh = feed_dict['mesh']

    return images, mesh

def compute_sampling_metrics(pred_points, gt_points, thresholds= [0.01, 0.02, 0.03, 0.04, 0.05], eps=1e-8):
    metrics = {}
    lengths_pred = torch.full(
        (pred_points.shape[0],), pred_points.shape[1], dtype=torch.int64, device=pred_points.device
    )
    lengths_gt = torch.full(
        (gt_points.shape[0],), gt_points.shape[1], dtype=torch.int64, device=gt_points.device
    )

    # For each predicted point, find its neareast-neighbor GT point
    knn_pred = knn_points(pred_points, gt_points, lengths1=lengths_pred, lengths2=lengths_gt, K=1)
    # Compute L1 and L2 distances between each pred point and its nearest GT
    pred_to_gt_dists2 = knn_pred.dists[..., 0]  # (N, S)
    pred_to_gt_dists = pred_to_gt_dists2.sqrt()  # (N, S)

    # For each GT point, find its nearest-neighbor predicted point
    knn_gt = knn_points(gt_points, pred_points, lengths1=lengths_gt, lengths2=lengths_pred, K=1)
    # Compute L1 and L2 dists between each GT point and its nearest pred point
    gt_to_pred_dists2 = knn_gt.dists[..., 0]  # (N, S)
    gt_to_pred_dists = gt_to_pred_dists2.sqrt()  # (N, S)

    # Compute precision, recall, and F1 based on L2 distances
    for t in thresholds:
        precision = 100.0 * (pred_to_gt_dists < t).float().mean(dim=1)
        recall = 100.0 * (gt_to_pred_dists < t).float().mean(dim=1)
        f1 = (2.0 * precision * recall) / (precision + recall + eps)
        metrics["Precision@%f" % t] = precision
        metrics["Recall@%f" % t] = recall
        metrics["F1@%f" % t] = f1

    # Move all metrics to CPU
    metrics = {k: v.cpu() for k, v in metrics.items()}
    return metrics

def evaluate(predictions, mesh_gt, args):
    if args.type == "vox" or args.type == "implicit":
        voxels_src = predictions
        H,W,D = voxels_src.shape[2:]
        vertices_src, faces_src = mcubes.marching_cubes(voxels_src.detach().cpu().squeeze().numpy(), isovalue=0.5)

        vertices_src = torch.tensor(vertices_src).float()
        faces_src = torch.tensor(faces_src.astype(int))

        if vertices_src.size(0) == 0:
            return None

        mesh_src = pytorch3d.structures.Meshes([vertices_src], [faces_src])
        pred_points = sample_points_from_meshes(mesh_src, args.n_points)
        pred_points = utils_vox.Mem2Ref(pred_points, H, W, D)
    elif args.type == "point":
        pred_points = predictions.cpu()
    elif args.type == "mesh":
        pred_points = sample_points_from_meshes(predictions, args.n_points).cpu()

    gt_points = sample_points_from_meshes(mesh_gt, args.n_points)

    metrics = compute_sampling_metrics(pred_points, gt_points)
    return metrics



def evaluate_model(args):
    r2n2_dataset = R2N2("test", dataset_location.SHAPENET_PATH, dataset_location.R2N2_PATH, dataset_location.SPLITS_PATH, return_voxels=True)

    loader = torch.utils.data.DataLoader(
        r2n2_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_batched_R2N2,
        pin_memory=True,
        drop_last=True)
    eval_loader = iter(loader)

    if args.type == 'implicit':
        model = ImplicitModel(args)
        coords = torch.arange(-1, 1, 2. / args.voxel_res) + 1. / args.voxel_res
        x, y, z = torch.meshgrid(coords, coords, coords)
        points = torch.stack((x, y, z), dim=-1).unsqueeze(0).cuda()
    else:
        model = SingleViewto3D(args)

    model.cuda()
    model.eval()

    start_iter = 0
    start_time = time.time()

    avg_f1_score = []

    if args.load_checkpoint:
        checkpoint = torch.load(f'checkpoint_{args.type}_{args.load_step}.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Succesfully loaded iter {start_iter}")

    print("Starting evaluating !")
    max_iter = len(eval_loader)
    for step in range(start_iter, max_iter):
        iter_start_time = time.time()

        read_start_time = time.time()

        feed_dict = next(eval_loader)

        images_gt, mesh_gt = preprocess(feed_dict)

        read_time = time.time() - read_start_time

        if args.type == 'implicit':
            predictions = model(images_gt, points, args)
        else:
            predictions = model(images_gt, args)

        if args.type == "vox":
            predictions = predictions.permute(0,1,4,3,2)

        metrics = evaluate(predictions, mesh_gt, args)

        if metrics is None:
            print("[%4d/%4d]; ttime: %.0f (%.2f, %.2f); Mesh Empty" % (step, max_iter, total_time, read_time, iter_time))
            continue

        # TODO:
        if (step % args.vis_freq) == 0:
            # visualization block
            # print(images_gt.shape)
            plt.imsave(f'vis/{step}_gt.png', images_gt.squeeze(0).cpu().detach().numpy())
            visualize_mesh(mesh_gt.cpu().detach(),
                          output_path=f'vis/{step}_gt.gif')

            if args.type == 'vox' or args.type == 'implicit':
                visualize_voxels_as_mesh(predictions[0, 0].cpu().detach(),
                                         output_path=f'vis/{step}_{args.type}.gif')
            elif args.type == 'point':
                visualize_point_cloud(predictions[0].cpu().detach(),
                                      output_path=f'vis/{step}_{args.type}.gif')
            elif args.type == 'mesh':
                visualize_mesh(predictions.cpu().detach(),
                               output_path=f'vis/{step}_{args.type}.gif')
            # plt.imsave(f'vis/{step}_{args.type}.png', rend)


        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        f1_05 = metrics['F1@0.050000']

        avg_f1_score.append(f1_05)

        print("[%4d/%4d]; ttime: %.0f (%.2f, %.2f); F1@0.05: %.3f; Avg F1@0.05: %.3f" % (step, max_iter, total_time, read_time, iter_time, f1_05, torch.tensor(avg_f1_score).mean()))

    print('Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Singleto3D', parents=[get_args_parser()])
    args = parser.parse_args()
    evaluate_model(args)
