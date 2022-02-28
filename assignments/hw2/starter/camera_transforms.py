"""
Usage:
    python -m starter.camera_transforms --image_size 512
"""
import argparse

import matplotlib.pyplot as plt
import pytorch3d
import torch

from pytorch3d.vis.plotly_vis import plot_scene
from starter.utils import get_device, get_mesh_renderer


def render_textured_cow(
    cow_path="data/cow.obj",
    image_size=256,
    R_relative=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    T_relative=[0, 0, 0],
    device=None,
):
    if device is None:
        device = get_device()


    if R_relative is None:
        R_relative = [[1., 0, 0], [0, 1, 0], [0, 0, 1]]

    if T_relative is None:
        T_relative = [0., 0, 0]

    meshes = pytorch3d.io.load_objs_as_meshes([cow_path]).to(device)
    R_relative = torch.tensor(R_relative).float()
    T_relative = torch.tensor(T_relative).float()
    R = R_relative @ torch.eye(3)
    T = R_relative @ torch.tensor([0.0, 0, 3]) + T_relative
    # T = T_relative
    renderer = get_mesh_renderer(image_size=256)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R.unsqueeze(0), T=T.unsqueeze(0), device=device,
    )

    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -3.0]],
                                            device=device)
    rend = renderer(meshes, cameras=cameras, lights=lights)
    return rend[0, ..., :3].cpu().numpy()

def render_all_cow_orientation(cow_path='data/cow.obj',
                               image_size=256,
                               output_path="output/textured_cow.jpg"):
    Rs_relative = [None,
                   [[0., 1, 0],
                    [-1, 0, 0],
                    [0, 0, 1]],
                   None,
                   [[0., 0., 1],
                    [0, 1, 0],
                    [-1., 0, 0]],]
    Ts_relative = [None,
                   None,
                   [0.5, -0.5, 0],
                   [-3, 0, 3]]

    for i, (R_relative, T_relative) in enumerate(zip(Rs_relative, Ts_relative)):
        plt.imsave('.'.join([output_path.split('.')[0] + '-{0}'.format(i),
                             output_path.split('.')[-1]]),
                   render_textured_cow(R_relative=R_relative,
                                       T_relative=T_relative))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cow_path", type=str, default="data/cow.obj")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--output_path", type=str, default="output/textured_cow.jpg")
    args = parser.parse_args()

    render_all_cow_orientation()
