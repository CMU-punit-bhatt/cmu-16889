import numpy as np
import argparse

import torch
from models import cls_model
from locality_models import PointNet2_Cls
from utils import create_dir, get_point_cloud_gif_360
from data_loader import get_data_loader

def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_cls_class', type=int, default=3, help='The number of classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='model_epoch_0')
    parser.add_argument('--i', type=int, default=0, help="index of the object to visualize")

    parser.add_argument('--test_data', type=str, default='/content/data/cls/data_test.npy')
    parser.add_argument('--test_label', type=str, default='/content/data/cls/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output')

    parser.add_argument('--exp_name', type=str, default="default", help='The name of the experiment - default, rotate, perturb')
    parser.add_argument('--perturb_scale', type=float, default=0.5, help='The perturbation scale')
    parser.add_argument('--rotation_angle', type=int, default=5, help='The rotation angle')

    # New
    parser.add_argument('--main_dir', type=str, default='/content/data/')
    parser.add_argument('--task', type=str, default="cls", help='The task: cls or seg')
    parser.add_argument('--batch_size', type=int, default=32, help='The number of images in a batch.')
    parser.add_argument('--num_workers', type=int, default=0, help='The number of threads to use for the DataLoader.')

    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    output_dir = f'./results/cls_locality/{args.exp_name}'

    # ------ TO DO: Initialize Model for Classification Task ------
    model = PointNet2_Cls(args)
    model = model.to(args.device)

    test_dataloader = get_data_loader(args=args, train=False)

    # Load Model Checkpoint
    model_path = './checkpoints_locality/cls/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print ("successfully loaded checkpoint from {}".format(model_path))

    # Sample Points per Object
    ind = np.random.choice(10000, args.num_points, replace=False)
    preds = []
    actual_labels = []

    correct_obj = 0
    num_obj = 0

    for i, batch in enumerate(test_dataloader):
        point_clouds, labels = batch
        point_clouds = point_clouds[:, ind].to(args.device)
        labels = labels.to(args.device).to(torch.long)

        if args.exp_name == 'perturb':

            # noise between - 0.5 and 0.5.
            noise = (torch.rand(point_clouds.shape).cuda() - 0.5) * args.perturb_scale
            point_clouds = point_clouds  + noise

        elif args.exp_name == 'rotate':
            rad = torch.Tensor([args.rotation_angle * np.pi / 180.])[0]

            R_x = torch.Tensor([[1, 0, 0],
                                [0, torch.cos(rad), - torch.sin(rad)],
                                [0, torch.sin(rad), torch.cos(rad)]])
            R_y = torch.Tensor([[torch.cos(rad), 0, torch.sin(rad)],
                                [0, 1, 0],
                                [- torch.sin(rad), 0, torch.cos(rad)]])
            R_z = torch.Tensor([[torch.cos(rad), - torch.sin(rad), 0],
                                [torch.sin(rad), torch.cos(rad), 0],
                                [0, 0, 1]])

            R = (R_x @ R_y @ R_z).cuda()

            point_clouds = torch.permute(R @ torch.permute(point_clouds,
                                                           (0, 2, 1)),
                                         (0, 2, 1))

        # ------ TO DO: Make Predictions ------
        with torch.no_grad():
            pred_labels = model(point_clouds)
            pred_labels = torch.argmax(torch.sigmoid(pred_labels), dim=-1)
        correct_obj += pred_labels.eq(labels.data).cpu().sum().item()
        num_obj += labels.size()[0]

        preds.extend(pred_labels.tolist())
        actual_labels.extend(labels.tolist())

    # Compute Accuracy of Test Dataset
    test_accuracy = correct_obj / num_obj
    test_data = torch.from_numpy((np.load(args.test_data))[:, ind, :])

    total_correct = 5
    total_incorrect = 5

    preds = torch.Tensor(preds)
    actual_labels = torch.Tensor(actual_labels)

    correct_inds = (preds == actual_labels).nonzero(as_tuple=True)[0]
    incorrect_inds = (preds != actual_labels).nonzero(as_tuple=True)[0]

    correct_inds = correct_inds[torch.randint(high=correct_inds.size(0),
                                size=(total_correct,))]
    incorrect_inds = incorrect_inds[torch.randint(high=incorrect_inds.size(0),
                                    size=(total_incorrect,))]

    print(preds[correct_inds], actual_labels[correct_inds])
    print(preds[incorrect_inds], actual_labels[incorrect_inds])

    if args.exp_name == 'perturb':
        output_dir = output_dir + f'_{args.perturb_scale}'
    
    elif args.exp_name == 'rotate':
        output_dir = output_dir + f'_{args.rotation_angle}'
    
    else:
        output_dir = output_dir + f'_{args.num_points}'

    create_dir(output_dir)
    
    for i in correct_inds:

        points = test_data[i]

        if args.exp_name == 'perturb':
            noise = (torch.rand(points.shape) - 0.5) * args.perturb_scale
            points = points + noise
        
        elif args.exp_name == 'rotate':
            rad = torch.Tensor([args.rotation_angle * np.pi / 180.])[0]

            R_x = torch.Tensor([[1, 0, 0],
                                [0, torch.cos(rad), - torch.sin(rad)],
                                [0, torch.sin(rad), torch.cos(rad)]])
            R_y = torch.Tensor([[torch.cos(rad), 0, torch.sin(rad)],
                                [0, 1, 0],
                                [- torch.sin(rad), 0, torch.cos(rad)]])
            R_z = torch.Tensor([[torch.cos(rad), - torch.sin(rad), 0],
                                [torch.sin(rad), torch.cos(rad), 0],
                                [0, 0, 1]])

            R = R_x @ R_y @ R_z

            points = (R @ points.T).T


        rgbs = torch.ones_like(points)

        get_point_cloud_gif_360(points,
                                rgbs,
                                output_path=f'{output_dir}/match_{i}_p{preds[i]}_l{actual_labels[i]}.gif')

    for i in incorrect_inds:

        points = test_data[i]

        if args.exp_name == 'perturb':
            noise = (torch.rand(points.shape) - 0.5) * args.perturb_scale
            points = points + noise
        
        elif args.exp_name == 'rotate':
            rad = torch.Tensor([args.rotation_angle * np.pi / 180.])[0]

            R_x = torch.Tensor([[1, 0, 0],
                                [0, torch.cos(rad), - torch.sin(rad)],
                                [0, torch.sin(rad), torch.cos(rad)]])
            R_y = torch.Tensor([[torch.cos(rad), 0, torch.sin(rad)],
                                [0, 1, 0],
                                [- torch.sin(rad), 0, torch.cos(rad)]])
            R_z = torch.Tensor([[torch.cos(rad), - torch.sin(rad), 0],
                                [torch.sin(rad), torch.cos(rad), 0],
                                [0, 0, 1]])

            R = R_x @ R_y @ R_z

            points = (R @ points.T).T

        rgbs = torch.ones_like(points)

        get_point_cloud_gif_360(points,
                                rgbs,
                                output_path=f'{output_dir}/mismatch_{i}_p{preds[i]}_l{actual_labels[i]}.gif')

    # Compute Accuracy
    print ("test accuracy: {}".format(test_accuracy))

