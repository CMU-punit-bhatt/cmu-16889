import argparse
import imageio
import matplotlib.pyplot as plt
import mcubes
import numpy as np
import pytorch3d
import torch
import pickle as pkl

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

def visualize(vertices, faces, color=[0.7, 0.7, 1], device=None):

    if device is None:
        device = get_device()

    if len(vertices.shape) < 3:
        vertices = vertices.unsqueeze(0)

    if len(faces.shape) < 3:
        faces = faces.unsqueeze(0)

    textures = torch.ones_like(vertices)  # (1, N_v, 3)
    textures = textures * torch.tensor(color)  # (1, N_v, 3)
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)
    # Better visualizer.
    fig = plot_scene({
        "subplot1": {
            "retexture_mesh": mesh,
        }
    })
    fig.show()

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

def dolly_zoom(
    image_size=256,
    num_frames=15,
    duration=3,
    device=None,
    output_file="output/dolly.gif",
):
    if device is None:
        device = get_device()

    mesh = pytorch3d.io.load_objs_as_meshes(["data/cow_on_plane.obj"])
    mesh = mesh.to(device)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    lights = pytorch3d.renderer.PointLights(location=[[0.0, 0.0, -3.0]],
                                            device=device)

    renders = []
    fovs = torch.linspace(5, 120, num_frames)
    init_dist = 5.
    distances = init_dist / (2 * torch.tan(torch.deg2rad(0.5 * fovs)))

    for i, fov in enumerate(tqdm(fovs)):
        T = [[0, 0, distances[i]]]
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(fov=fov,
                                                           T=T,
                                                           device=device)
        rend = renderer(mesh, cameras=cameras, lights=lights)
        rend = rend[0, ..., :3].cpu().numpy()
        renders.append(rend)

    images = []
    for i, r in enumerate(renders):
        image = Image.fromarray((r * 255).astype(np.uint8))
        draw = ImageDraw.Draw(image)
        draw.text((20, 20), f"fov: {fovs[i]:.2f}", fill=(255, 0, 0))
        images.append(np.array(image))

    imageio.mimsave(output_file, images, fps=(num_frames / duration))

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

def construct_tetrahedron(output_path='output/tetrahedron.gif',
                          image_size=256,
                          color=[0.7, 0.7, 1],
                          device=None):

    if device is None:
        device = get_device()

    base = 10
    base_half = base / 2
    height = 10

    verts = torch.tensor([
        [- base_half, - height / 2, - base_half],
        [base_half, - height / 2, - base_half],
        [0, - height / 2, base_half],
        [0, height / 2, 0],
    ]).to(device)

    # Assuming counter-clockwise is outwards orientation.
    faces = torch.tensor([
        [0, 3, 1],
        [1, 3, 2],
        [2, 3, 0],
        [0, 1, 2],
    ]).to(device)

    verts = verts.unsqueeze(0)
    faces = faces.unsqueeze(0)

    # visualize(verts, faces)

    get_mesh_gif_360(verts,
                     faces,
                     output_path=output_path,
                     image_size=image_size,
                     distance=15,
                     fov=100,
                     color=color)

def construct_cube(output_path='output/cube.gif',
                   image_size=256,
                   color=[0.7, 0.7, 1]):

    device = get_device()
    base = 10
    base_half = base / 2

    verts = torch.tensor([
        #Bottom 4
        [- base_half, - base_half, base_half],
        [base_half, - base_half, base_half],
        [base_half, - base_half, - base_half],
        [- base_half, - base_half, - base_half],
        #Top4
        [- base_half, base_half, base_half],
        [base_half, base_half, base_half],
        [base_half, base_half, - base_half],
        [- base_half, base_half, - base_half],
    ]).to(device)

    # Assuming counter-clockwise is outwards orientation.
    faces = torch.tensor([
        [0, 1, 4],
        [1, 5, 4],
        [1, 2, 5],
        [2, 6, 5],
        [2, 3, 6],
        [3, 7, 6],
        [3, 0, 7],
        [0, 4, 7],
        [0, 1, 3],
        [1, 2, 3],
        [4, 5, 7],
        [5, 6, 7]
    ]).to(device)

    verts = verts.unsqueeze(0)
    faces = faces.unsqueeze(0)

    # visualize(verts, faces)

    get_mesh_gif_360(verts,
                     faces,
                     output_path=output_path,
                     image_size=image_size,
                     distance=15,
                     fov=100,
                     color=color)

def retextured_mesh(vertices,
                    faces,
                    color1=[0, 0, 1],
                    color2=[1, 0, 0],
                    output_path='output/retextured_mesh.gif',
                    device=None):

    if device is None:
        device = get_device()

    z = vertices[:, -1]
    alpha = (z - torch.min(z)) / (torch.max(z) - torch.min(z))
    alpha = alpha.repeat(3, 1).T

    color1 = torch.tensor(color1)
    color2 = torch.tensor(color2)

    color = alpha * color2 + (1 - alpha) * color1

    assert color.shape[0] == vertices.shape[0]

    textures = torch.ones_like(vertices)
    textures = textures * color

    get_mesh_gif_360(vertices, faces, textures=textures, output_path=output_path)

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
    Rs_relative = [[[0., 1, 0],
                    [-1, 0, 0],
                    [0, 0, 1]],
                   None,
                   None,
                   [[0., 0., 1],
                    [0, 1, 0],
                    [-1., 0, 0]],]
    Ts_relative = [None,
                   [0, 0, 2.],
                   [0.5, -0.5, 0],
                   [-3., 0, 3.]]

    for i, (R_relative, T_relative) in enumerate(zip(Rs_relative, Ts_relative)):
        plt.imsave('.'.join([output_path.split('.')[0] + '-{0}'.format(i),
                             output_path.split('.')[-1]]),
                   render_textured_cow(R_relative=R_relative,
                                       T_relative=T_relative))

def render_point_cloud(pickle_file='data/rgbd_data.pkl',
                       output_path='output/point_cloud.gif',
                       device=None):

    if device is None:
        device = get_device()

    with open(pickle_file, 'rb') as f:
        dict = pkl.load(f)

    pts1, rgba1 = unproject_depth_image(torch.from_numpy(dict['rgb1']),
                                       torch.from_numpy(dict['mask1']),
                                       torch.from_numpy(dict['depth1']),
                                       dict['cameras1'])

    pts2, rgba2 = unproject_depth_image(torch.from_numpy(dict['rgb2']),
                                       torch.from_numpy(dict['mask2']),
                                       torch.from_numpy(dict['depth2']),
                                       dict['cameras2'])

    # Rotating points to get the rightside up
    R_invert = torch.tensor([[-1., 0, 0],
                             [0, -1, 0],
                             [0, 0, 1]])
    pts1_inverted = pts1 @ R_invert
    pts2_inverted = pts2 @ R_invert

    pts_union_inverted = torch.cat((pts1_inverted, pts2_inverted), dim=0)
    rgba_union_inverted = torch.cat((rgba1, rgba2), dim=0)

    get_point_cloud_gif_360(pts1_inverted,
                            rgba1[:, : -1],
                            output_path='output/plant1.gif')
    get_point_cloud_gif_360(pts2_inverted,
                            rgba2[:, : -1],
                            output_path='output/plant2.gif')
    get_point_cloud_gif_360(pts_union_inverted,
                            rgba_union_inverted[:, : -1],
                            output_path='output/plant_union.gif')

def render_torus_parametric(hole_radius=2,
                            tube_radius=1,
                            image_size=256,
                            num_samples=200,
                            device=None):
    """
    Renders a sphere using parametric sampling. Samples num_samples ** 2 points.
    """

    if device is None:
        device = get_device()

    torus_R = hole_radius + tube_radius
    torus_r = tube_radius

    phi = torch.linspace(0, 2 * np.pi, num_samples)
    theta = torch.linspace(0, 2 * np.pi, num_samples)
    Phi, Theta = torch.meshgrid(phi, theta)

    x = (torus_R + torus_r * torch.cos(Theta)) * torch.cos(Phi)
    y = (torus_R + torus_r * torch.cos(Theta)) * torch.sin(Phi)
    z = torus_r * torch.sin(Theta)

    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())

    get_point_cloud_gif_360(points,
                            color,
                            image_size=image_size,
                            output_path='output/torus_{0}.gif'.format(
                                num_samples))

def render_torus_mesh(hole_radius=2,
                      tube_radius=1,
                      image_size=256,
                      voxel_size=64,
                      device=None):

    if device is None:
        device = get_device()

    torus_R = hole_radius + tube_radius
    torus_r = tube_radius

    min_value = - (torus_R + torus_r + 0.1)
    max_value = (torus_R + torus_r + 0.1)
    X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, voxel_size)] * 3)
    voxels = ((X ** 2 + Y ** 2) ** 0.5 - torus_R) ** 2 + Z ** 2 - torus_r ** 2
    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))

    # Vertex coordinates are indexed by array position, so we need to
    # renormalize the coordinate system.
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())

    get_mesh_gif_360(vertices,
                     faces,
                     textures=textures,
                     distance=torus_R + torus_r + 5.,
                     output_path='output/torus_mesh.gif')

def something_fun(device=None,
                  output_path='output/ufo.gif',
                  fov=60,
                  image_size=512):

    if device is None:
        device = get_device()

    mesh_ufo = pytorch3d.io.load_objs_as_meshes(
        ["data/ufo/13884_UFO_Saucer_v1_l2.obj"])
    mesh_alien = pytorch3d.io.load_objs_as_meshes(
        ["data/alien/10469_GrayAlien_v01.obj"])

    steps_ufo = range(475, 50, -25)
    steps_alien = range(90, -110, -5)

    R_ufo = torch.tensor([[1., 0, 0], [0, 0, 1], [0, -1, 0]])
    distance_ufo = 150.

    R_alien = torch.tensor([[-1., 0, 0], [0, 0, 1], [0, 1, 0.]])
    distance_alien = 250.

    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)

    # Place a point light in front of the cow.
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]],
                                            device=device)

    images = []

    R = R_ufo.unsqueeze(0)

    print('Starting UFO rendering...')

    for i in tqdm(steps_ufo):

        T = torch.tensor([0, distance_ufo, i]).unsqueeze(0)

        # Prepare the camera:
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R,
                                                           T=T,
                                                           fov=fov,
                                                           device=device)

        rend = renderer(mesh_ufo, cameras=cameras, lights=lights)
        images.append((rend.cpu().numpy()[0, ..., :3] * 255).astype(np.uint8))

    print('Starting ALIEN rendering...')

    R = R_alien.unsqueeze(0)

    for i in tqdm(steps_alien):

        T = torch.tensor([0, i, distance_alien]).unsqueeze(0)

        # Prepare the camera:
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R,
                                                           T=T,
                                                           fov=fov,
                                                           device=device)

        rend = renderer(mesh_alien, cameras=cameras, lights=lights)
        images.append((rend.cpu().numpy()[0, ..., :3] * 255).astype(np.uint8))

    generate_gif(images, output_path)

def point_cloud_from_mesh(num_samples=100,
                          output_path='output/cow_point_cloud.gif',
                          color = [0.7, 0.7, 0.7],
                          device=None):

    if device is None:
        device = get_device()

    mesh = pytorch3d.io.load_objs_as_meshes(["data/cow.obj"])

    vertices = mesh.verts_packed()
    faces = mesh.faces_packed()
    areas = mesh.faces_areas_packed()
    face_to_sample = torch.multinomial(areas, num_samples, replacement=True)

    alpha1 = 1. - torch.rand(num_samples)
    alpha2 = (1. - alpha1) * torch.rand(num_samples)
    alpha3 = (1. - alpha1) * (1. - alpha2)

    alpha1 = alpha1.reshape(-1, 1)
    alpha2 = alpha2.reshape(-1, 1)
    alpha3 = alpha3.reshape(-1, 1)

    vertices_to_sample = faces[face_to_sample]
    x = vertices[vertices_to_sample[:, 0]]
    y = vertices[vertices_to_sample[:, 1]]
    z = vertices[vertices_to_sample[:, 2]]

    points = alpha1 * x + alpha2 * y + alpha3 * z

    rgbs = torch.ones_like(points) * torch.tensor(color)

    output_path = '.'.join([output_path.split('.')[0] + '-{0}'.format(num_samples),
                            output_path.split('.')[-1]])
    get_point_cloud_gif_360(points, rgbs, distance=5.0, output_path=output_path)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-q',
                        '--question',
                        help="Defines the question number.",
                        default='1.1')
    parser.add_argument('-n',
                        '--num-samples',
                        help="Defines the number of samples.",
                        default=100)
    args = parser.parse_args()
    question = args.question
    num_samples = int(args.num_samples)

    if question == '1.1':
        vertices, faces = load_cow_mesh('data/cow.obj')
        get_mesh_gif_360(vertices, faces)

    elif question == '1.2':
        dolly_zoom(image_size=256,
                   num_frames=15,
                   duration=5,
                   output_file='output/cow_dolly_zoom.gif')

    elif question == '2.1':
        construct_tetrahedron()

    elif question == '2.2':
        construct_cube()

    elif question == '3':
        vertices, faces = load_cow_mesh('data/cow.obj')
        retextured_mesh(vertices,
                        faces,
                        color1=[0, 0.6, 0.6],
                        color2=[0.8, 0, 0.4])

    elif question == '4':
        render_all_cow_orientation()

    elif question == '5.1':
        render_point_cloud()

    elif question == '5.2':
        render_torus_parametric(num_samples=num_samples)

    elif question == '5.3':
        render_torus_mesh()

    elif question == '6':
        something_fun()

    elif question == '7':
        point_cloud_from_mesh(num_samples=num_samples)

    else:
        print('Invalid question!')