import imageio
import matplotlib.pyplot as plt
import mcubes
import numpy as np
import pytorch3d
import torch

from PIL import Image, ImageDraw
from pytorch3d.vis.plotly_vis import plot_scene
from pytorch3d.renderer.blending import BlendParams
from starter.utils import (
    get_device,
    get_mesh_renderer,
    load_cow_mesh,
    get_points_renderer,
    unproject_depth_image
)
from tqdm.auto import tqdm


def generate_gif(images, path):
    imageio.mimsave(path, images, fps=15)

def get_mesh_gif_360(vertices,
                     faces,
                     textures=None,
                     output_path='mygif.gif',
                     distance=3.0,
                     fov=60,
                     image_size=256,
                     color=[0.7, 0.7, 1],
                     steps=range(360, 0, -15)):

    device = get_device()

    if len(vertices.shape) < 3:
        vertices = vertices.unsqueeze(0)

    if len(faces.shape) < 3:
        faces = faces.unsqueeze(0)

    if textures is None:
        # Get the vertices, faces, and textures.
        textures = torch.ones_like(vertices)  # (1, N_v, 3)
        textures = textures * torch.tensor(color)  # (1, N_v, 3)
    elif len(textures.shape) < 3:
        textures = textures.unsqueeze(0)

    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)

    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)

    # Place a point light in front of the cow.
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]],
                                            device=device)

    images = []

    for i in tqdm(steps):

        # print(f'{i = }')

        R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist=distance,
                                                                 azim=i,
                                                                 device=device)

        # Prepare the camera:
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R,
                                                           T=T,
                                                           fov=fov,
                                                           device=device)

        rend = renderer(mesh, cameras=cameras, lights=lights)
        images.append((rend.cpu().numpy()[0, ..., :3] * 255).astype(np.uint8))

    generate_gif(images, output_path)

def get_point_cloud_gif_360(verts,
                            rgb,
                            output_path='mygif.gif',
                            distance=10.0,
                            fov=60,
                            image_size=256,
                            background_color=[1., 1, 1],
                            steps=range(360, 0, -15)):

    device = get_device()

    if len(verts.shape) < 3:
        verts = verts.unsqueeze(0)

    if len(rgb.shape) < 3:
        rgb = rgb.unsqueeze(0)

    renderer = get_points_renderer(
        image_size=image_size, background_color=background_color
    )
    point_cloud = pytorch3d.structures.Pointclouds(points=verts,
                                                   features=rgb).to(device)

    # Place a point light in front of the cow.
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]],
                                            device=device)

    images = []

    for i in tqdm(steps):

        # print(f'{i = }')

        R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist=distance,
                                                                 azim=i,
                                                                 device=device)
        # Prepare the camera:
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R,
                                                           T=T,
                                                           fov=fov,
                                                           device=device)

        rend = renderer(point_cloud, cameras=cameras)
        images.append((rend.cpu().numpy()[0, ..., :3] * 255).astype(np.uint8))

    return generate_gif(images, output_path)

def visualize_voxels_as_mesh(voxels,
                             voxel_size=32,
                             image_size=256,
                             distance=3.0,
                             fov=60,
                             output_path='mygif.gif',
                             device=None):

    if device is None:
      device = get_device()

    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))

    max_value = 1
    min_value = -1

    # Vertex coordinates are indexed by array position, so we need to
    # renormalize the coordinate system.
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())

    # vertices = vertices.unsqueeze(0)
    # faces = faces.unsqueeze(0)
    # textures = textures.unsqueeze(0)

    # mesh = pytorch3d.structures.Meshes(
    #     verts=vertices,
    #     faces=faces,
    #     textures=pytorch3d.renderer.TexturesVertex(textures),
    # )
    # mesh = mesh.to(device)

    # # Get the renderer.
    # renderer = get_mesh_renderer(image_size=image_size)

    # # Place a point light in front of the cow.
    # lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]],
    #                                         device=device)
    # R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist=distance,
    #                                                          azim=0,
    #                                                          device=device)

    # # Prepare the camera:
    # cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R,
    #                                                    T=T,
    #                                                    fov=fov,
    #                                                    device=device)
    # rend = renderer(mesh, cameras=cameras, lights=lights)
    # plt.imshow((rend.cpu().numpy()[0, ..., :3] * 255).astype(np.uint8))
    # plt.savefig(output_path)

    get_mesh_gif_360(vertices,
                     faces,
                     textures=textures,
                     distance=distance,
                     fov=fov,
                     output_path=output_path)

def visualize_point_cloud(points,
                          image_size=256,
                          color=[0.7, 0.7, 1],
                          distance=3.0,
                          fov=60,
                          output_path='mygif.gif',
                          device=None):

    if device is None:
      device = get_device()
    
    rgbs = torch.ones_like(points) * torch.tensor(color)

    get_point_cloud_gif_360(points,
                            rgbs,
                            distance=distance,
                            fov=fov,
                            image_size=image_size,
                            output_path=output_path)

                
def visualize_mesh(mesh,
                   image_size=256,
                   distance=3.0,
                   fov=60,
                   output_path='mygif.gif',
                   device=None):

    vertices = mesh.verts_packed()
    faces = mesh.faces_packed() 

    get_mesh_gif_360(vertices,
                     faces, 
                     output_path=output_path, 
                     distance=distance,
                     fov=fov,
                     image_size=image_size)
