import numpy as np
import argparse

import torch
from models import seg_model
from data_loader import get_data_loader
from utils import create_dir, viz_seg


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_seg_class', type=int, default=6, help='The number of segmentation classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='model_epoch_0')
    parser.add_argument('--i', type=int, default=0, help="index of the object to visualize")

    parser.add_argument('--test_data', type=str, default='/content/data/seg/data_test.npy')
    parser.add_argument('--test_label', type=str, default='/content/data/seg/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output')

    parser.add_argument('--exp_name', type=str, default="default", help='The name of the experiment - default, rotate, perturb')
    parser.add_argument('--perturb_scale', type=float, default=0.5, help='The perturbation scale')
    parser.add_argument('--rotation_angle', type=int, default=5, help='The rotation angle')
    parser.add_argument('--bad_accuracy', type=float, default=0.6, help='The accuracy below which predictions are considered bad.')

    parser.add_argument('--main_dir', type=str, default='/content/data/')
    parser.add_argument('--task', type=str, default="seg", help='The task: cls or seg')
    parser.add_argument('--batch_size', type=int, default=32, help='The number of images in a batch.')
    parser.add_argument('--num_workers', type=int, default=0, help='The number of threads to use for the DataLoader.')

    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    output_dir = f'./results/seg/{args.exp_name}'

    # ------ TO DO: Initialize Model for Segmentation Task  ------
    model = seg_model()
    model = model.to(args.device)

    test_dataloader = get_data_loader(args=args, train=False)
    
    # Load Model Checkpoint
    model_path = './checkpoints/seg/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print ("successfully loaded checkpoint from {}".format(model_path))


    # Sample Points per Object
    ind = np.random.choice(10000,args.num_points, replace=False)

    correct_point = 0
    num_point = 0

    preds = []
    actual_labels = []

    for batch in test_dataloader:
        point_clouds, labels = batch
        point_clouds = point_clouds.to(args.device)[:,ind]
        labels = labels.to(args.device).to(torch.long)[:,ind]

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
            
        correct_point += pred_labels.eq(labels.data).cpu().sum().item()
        num_point += labels.view([-1,1]).size()[0]

        preds.extend(pred_labels.tolist())
        actual_labels.extend(labels.tolist())

    # Compute Accuracy of Test Dataset
    test_accuracy = correct_point / num_point

    test_data = torch.from_numpy((np.load(args.test_data))[:, ind, :])

    total_correct = 5
    total_incorrect = 5

    preds = torch.Tensor(preds)
    actual_labels = torch.Tensor(actual_labels)

    accuracies = torch.sum(preds == actual_labels, dim=-1) / actual_labels.size(-1)

    correct_inds = (accuracies >= 0.85).nonzero(as_tuple=True)[0]
    incorrect_inds = (accuracies <= args.bad_accuracy).nonzero(as_tuple=True)[0]

    try:
        correct_inds = correct_inds[torch.randint(high=correct_inds.size(0),
                                                  size=(total_correct,))]
    except:
        correct_inds = []
    
    try:
        incorrect_inds = incorrect_inds[torch.randint(high=incorrect_inds.size(0),
                                                      size=(total_incorrect,))]
    except:
        incorrect_inds = []

    print(accuracies[correct_inds])
    print(accuracies[incorrect_inds])

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
        
        viz_seg(points, 
                actual_labels[i],
                f"{output_dir}/{i}_{round(accuracies[i].item(), 3)}_match_gt.gif", 
                args.device,
                args.num_points)

        viz_seg(points, 
                preds[i],
                f"{output_dir}/{i}_{round(accuracies[i].item(), 3)}_match_pred.gif", 
                args.device,
                args.num_points)

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

        viz_seg(points, 
                actual_labels[i],
                f"{output_dir}/{i}_{round(accuracies[i].item(), 3)}_mismatch_gt.gif", 
                args.device,
                args.num_points)

        viz_seg(points, 
                preds[i],
                f"{output_dir}/{i}_{round(accuracies[i].item(), 3)}_mismatch_pred.gif", 
                args.device,
                args.num_points)

    print ("test accuracy: {}".format(test_accuracy))
