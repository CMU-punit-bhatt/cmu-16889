import numpy as np
import torch
import torch.nn.functional as F

from torch import autograd

from ray_utils import RayBundle


# Sphere SDF class
class SphereSDF(torch.nn.Module):
    def __init__(
        self,
        cfg,
        center=None,
        radius=None
    ):
        super().__init__()

        center_grad = False
        radius_grad = False

        if center is None:
            center = cfg.center.val
            center_grad = cfg.center.opt

        self.center = torch.nn.Parameter(
            torch.tensor(center).float().unsqueeze(0), requires_grad=center_grad
        )

        if radius is None:
            radius = cfg.radius.val
            radius_grad = cfg.radius.opt

        self.radius = torch.nn.Parameter(
            torch.tensor(radius).float(), requires_grad=radius_grad
        )

    def forward(self, points):
        points = points.view(-1, 3)

        return torch.linalg.norm(
            points - self.center,
            dim=-1,
            keepdim=True
        ) - self.radius


# Box SDF class
class BoxSDF(torch.nn.Module):
    def __init__(
        self,
        cfg,
        center=None,
        side=None
    ):
        super().__init__()

        center_grad = False
        side_grad = False

        if center is None:
            center = cfg.center.val
            center_grad = cfg.center.opt

        self.center = torch.nn.Parameter(
            torch.tensor(center).float().unsqueeze(0), requires_grad=center_grad
        )

        if side is None:
            side = cfg.side_lengths.val
            side_grad = cfg.side_lengths.opt

        self.center = torch.nn.Parameter(
            torch.tensor(center).float().unsqueeze(0), requires_grad=center_grad
        )
        self.side_lengths = torch.nn.Parameter(
            torch.tensor(side).float().unsqueeze(0), requires_grad=side_grad
        )

    def forward(self, points):
        points = points.view(-1, 3)
        diff = torch.abs(points - self.center) - self.side_lengths / 2.0

        signed_distance = torch.linalg.norm(
            torch.maximum(diff, torch.zeros_like(diff)),
            dim=-1
        ) + torch.minimum(torch.max(diff, dim=-1)[0], torch.zeros_like(diff[..., 0]))

        return signed_distance.unsqueeze(-1)

# Torus SDF class
class TorusSDF(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.center = torch.nn.Parameter(
            torch.tensor(cfg.center.val).float().unsqueeze(0), requires_grad=cfg.center.opt
        )
        self.radii = torch.nn.Parameter(
            torch.tensor(cfg.radii.val).float().unsqueeze(0), requires_grad=cfg.radii.opt
        )

    def forward(self, points):
        points = points.view(-1, 3)
        diff = points - self.center
        q = torch.stack(
            [
                torch.linalg.norm(diff[..., :2], dim=-1) - self.radii[..., 0],
                diff[..., -1],
            ],
            dim=-1
        )
        return (torch.linalg.norm(q, dim=-1) - self.radii[..., 1]).unsqueeze(-1)
        
# Marvel Cube SDF class
class MarvelCubeSDF(torch.nn.Module):
    def __init__(
        self,
        cfg,
        offset=[0, 0, 0]
    ):
        super().__init__()

        self.offset = torch.Tensor(offset).cuda()

        # Inner hollow sphere with a cube in it.
        self.inner_sphere1 = SphereSDF(cfg,
                                       torch.Tensor(cfg.inner_center.val).cuda() + \
                                          self.offset,
                                       cfg.inner_radius1.val)
        self.inner_sphere2 = SphereSDF(cfg,
                                       torch.Tensor(cfg.inner_center.val).cuda() + \
                                          self.offset,
                                       cfg.inner_radius2.val)
        self.inner_cube = BoxSDF(cfg,
                                 torch.Tensor(cfg.inner_center.val).cuda() + \
                                     self.offset,
                                 cfg.inner_side.val)
        
        # Holes in the hollow sphere.
        self.indent_cube1 = BoxSDF(cfg,
                                   torch.Tensor(cfg.indent_center1.val).cuda() + self.offset, 
                                   cfg.indent_side1.val)
        self.indent_cube2 = BoxSDF(cfg,
                                   torch.Tensor(cfg.indent_center2.val).cuda() + self.offset, 
                                   cfg.indent_side2.val)
        self.indent_cube3 = BoxSDF(cfg,
                                   torch.Tensor(cfg.indent_center3.val).cuda() + self.offset, 
                                   cfg.indent_side3.val)
        self.indent_cube4 = BoxSDF(cfg,
                                   torch.Tensor(cfg.indent_center4.val).cuda() + self.offset, 
                                   cfg.indent_side4.val)
        self.indent_cube5 = BoxSDF(cfg,
                                   torch.Tensor(cfg.indent_center5.val).cuda() + self.offset, 
                                   cfg.indent_side5.val)
        self.indent_cube6 = BoxSDF(cfg,
                                   torch.Tensor(cfg.indent_center6.val).cuda() + self.offset, 
                                   cfg.indent_side6.val)
        
        # External box
        self.ext_cube1 = BoxSDF(cfg,
                                torch.Tensor(cfg.inner_center.val).cuda() + self.offset, 
                                cfg.ext_side1.val)
        self.ext_cube2 = BoxSDF(cfg,
                                torch.Tensor(cfg.inner_center.val).cuda() + self.offset, 
                                cfg.ext_side2.val)
        self.ext_indent_cube1 = BoxSDF(cfg,
                                       torch.Tensor(cfg.ext_indent_center1.val).cuda() + self.offset, 
                                       cfg.ext_indent_side.val)
        self.ext_indent_cube2 = BoxSDF(cfg,
                                       torch.Tensor(cfg.ext_indent_center2.val).cuda() + self.offset, 
                                       cfg.ext_indent_side.val)
        self.ext_indent_cube3 = BoxSDF(cfg,
                                       torch.Tensor(cfg.ext_indent_center3.val).cuda() + self.offset, 
                                       cfg.ext_indent_side.val)
        self.ext_indent_cube4 = BoxSDF(cfg,
                                       torch.Tensor(cfg.ext_indent_center4.val).cuda() + self.offset, 
                                       cfg.ext_indent_side.val)
        self.ext_indent_cube5 = BoxSDF(cfg,
                                       torch.Tensor(cfg.ext_indent_center5.val).cuda() + self.offset, 
                                       cfg.ext_indent_side.val)
        self.ext_indent_cube6 = BoxSDF(cfg,
                                       torch.Tensor(cfg.ext_indent_center6.val).cuda() + self.offset, 
                                       cfg.ext_indent_side.val)

        # External Ring
        self.ring_sphere1 = SphereSDF(cfg,
                                      torch.Tensor(cfg.inner_center.val).cuda() + self.offset, 
                                      cfg.ring_sphere_radius1.val)
        self.ring_sphere2 = SphereSDF(cfg,
                                      torch.Tensor(cfg.inner_center.val).cuda() + self.offset, 
                                      cfg.ring_sphere_radius2.val)
        self.ring_cube1 = BoxSDF(cfg,
                                 torch.Tensor(cfg.ring_cube_center1.val).cuda() + self.offset, 
                                 cfg.ring_cube_side.val)
        self.ring_cube2 = BoxSDF(cfg,
                                 torch.Tensor(cfg.ring_cube_center2.val).cuda() + self.offset, 
                                 cfg.ring_cube_side.val)

        self.inner_cube_color = torch.Tensor([1., 0.549, 0.]).cuda()
        self.inner_sphere_color = torch.Tensor([0.831, 0.686, 0.216]).cuda()
        self.external_box_color = torch.Tensor([0.30980392, 0.4745098 , 0.25882353]).cuda()
        self.ring_color = torch.Tensor([1., 0.40392157, 0.]).cuda()

    def forward(self, points):
        points = points.view(-1, 3)

        dists_sphere1 = self.inner_sphere1(points)
        dists_sphere2 = self.inner_sphere2(points)
        dists_cube = self.inner_cube(points)
        dists1 = self.indent_cube1(points)
        dists2 = self.indent_cube2(points)
        dists3 = self.indent_cube3(points)
        dists4 = self.indent_cube4(points)
        dists5 = self.indent_cube5(points)
        dists6 = self.indent_cube6(points)

        ring_sphere_dists1 = self.ring_sphere1(points)
        ring_sphere_dists2 = self.ring_sphere2(points)
        ring_cube_dists1 = self.ring_cube1(points)
        ring_cube_dists2 = self.ring_cube2(points)

        ext_dists1 = self.ext_cube1(points)
        ext_dists2 = self.ext_cube2(points)

        ext_indent_dists1 = self.ext_indent_cube1(points)
        ext_indent_dists2 = self.ext_indent_cube2(points)
        ext_indent_dists3 = self.ext_indent_cube3(points)
        ext_indent_dists4 = self.ext_indent_cube4(points)
        ext_indent_dists5 = self.ext_indent_cube5(points)
        ext_indent_dists6 = self.ext_indent_cube6(points)

        ring_dists = torch.amax(torch.hstack((ring_sphere_dists1,
                                              - ring_sphere_dists2
                                              - ring_cube_dists1,
                                              - ring_cube_dists2)),
                                dim=-1,
                                keepdim=True)

        dists = torch.amax(torch.hstack((ext_dists1,
                                         - ext_dists2, 
                                         - ext_indent_dists1, 
                                         - ext_indent_dists2, 
                                         - ext_indent_dists3, 
                                         - ext_indent_dists4, 
                                         - ext_indent_dists5, 
                                         - ext_indent_dists6)),
                           dim=-1,
                           keepdims=True)

        dists = torch.minimum(ring_dists, dists)

        dists = torch.minimum(dists, dists_sphere1)
        dists = torch.amax(torch.hstack((dists,
                                         - dists_sphere2,
                                         - dists1,
                                         - dists2, 
                                         - dists3, 
                                         - dists4, 
                                         - dists5, 
                                         - dists6)),
                           dim=-1,
                           keepdims=True)
        dists = torch.minimum(dists, dists_cube)

        return dists
    
    def get_textures(self, points):
      
        eps = 1e-3
        base_color = torch.ones_like(points).cuda() * \
            torch.Tensor([0.03529412, 0.4745098 , 0.41176471]).cuda()
        mask1 = torch.torch.linalg.norm(points - self.offset, dim=-1) < \
            self.inner_sphere1.radius - eps
        mask2 = torch.torch.linalg.norm(points - self.offset, dim=-1) < \
            self.inner_sphere2.radius - eps
        mask3 = torch.torch.linalg.norm(points - self.offset, dim=-1) > \
            self.inner_sphere1.radius + eps
        mask4 = torch.torch.linalg.norm(points - self.offset, dim=-1) >= \
            self.ring_sphere2.radius - 0.2
        base_color[mask1] = torch.Tensor([0.831, 0.686, 0.216]).cuda()
        base_color[mask2] = torch.Tensor([1., 0.549, 0.]).cuda()
        base_color[mask3] = torch.Tensor([0.30980392, 0.4745098 , 0.25882353]).cuda()
        base_color[mask4] = torch.Tensor([1., 0.40392157, 0.]).cuda()

        return base_color

# Trippy class
class TrippySDF(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.offset_val = 4.5

        self.marvel_cube1 = MarvelCubeSDF(cfg, offset=[self.offset_val, 0, 0])
        self.marvel_cube2 = MarvelCubeSDF(cfg, offset=[- self.offset_val, 0, 0])
        self.marvel_cube3 = MarvelCubeSDF(cfg, offset=[0, self.offset_val, 0])
        self.marvel_cube4 = MarvelCubeSDF(cfg, offset=[0, - self.offset_val, 0])
        self.marvel_cube5 = MarvelCubeSDF(cfg, offset=[0, 0, 0])

    def forward(self, points):
        points = points.view(-1, 3)
        
        dists1 = self.marvel_cube1(points)
        dists2 = self.marvel_cube2(points)
        dists3 = self.marvel_cube3(points)
        dists4 = self.marvel_cube4(points)
        dists5 = self.marvel_cube5(points)

        dists = torch.amin(torch.hstack((dists1,
                                         dists2, 
                                         dists3, 
                                         dists4,
                                         dists5)),
                           dim=-1,
                           keepdims=True)

        return dists
    
    def get_textures(self, points):

        texts1 = self.marvel_cube1.get_textures(points)
        texts2 = self.marvel_cube2.get_textures(points)
        texts3 = self.marvel_cube3.get_textures(points)
        texts4 = self.marvel_cube4.get_textures(points)
        texts5 = self.marvel_cube5.get_textures(points)

        base_color = texts5

        eps = 1e-3
        mask1 = torch.torch.linalg.norm(points - self.marvel_cube1.offset,
                                        dim=-1) <= self.marvel_cube1.ring_sphere1.radius + eps
        mask2 = torch.torch.linalg.norm(points - self.marvel_cube2.offset, 
                                        dim=-1) <= self.marvel_cube2.ring_sphere1.radius + eps
        mask3 = torch.torch.linalg.norm(points - self.marvel_cube3.offset, 
                                        dim=-1) <= self.marvel_cube3.ring_sphere1.radius + eps
        mask4 = torch.torch.linalg.norm(points - self.marvel_cube4.offset, 
                                        dim=-1) <= self.marvel_cube4.ring_sphere1.radius + eps

        base_color[mask1] = texts1[mask1]
        base_color[mask2] = texts2[mask2]
        base_color[mask3] = texts3[mask3]
        base_color[mask4] = texts4[mask4]

        return base_color

sdf_dict = {
    'sphere': SphereSDF,
    'box': BoxSDF,
    'torus': TorusSDF,
    'marvel': MarvelCubeSDF,
    'trippy': TrippySDF
}


# Converts SDF into density/feature volume
class SDFSurface(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.sdf = sdf_dict[cfg.sdf.type](
            cfg.sdf
        )
        self.rainbow = cfg.feature.rainbow if 'rainbow' in cfg.feature else False
        self.marvel = cfg.feature.marvel if 'marvel' in cfg.feature else False
        self.trippy = cfg.feature.trippy if 'trippy' in cfg.feature else False
        self.feature = torch.nn.Parameter(
            torch.ones_like(torch.tensor(cfg.feature.val).float().unsqueeze(0)), requires_grad=cfg.feature.opt
        )
    
    def get_distance(self, points):
        points = points.view(-1, 3)
        return self.sdf(points)

    def get_color(self, points):
        points = points.view(-1, 3)

        # Outputs
        if self.rainbow:
            base_color = torch.clamp(
                torch.abs(points - self.sdf.center),
                0.02,
                0.98
            )
        elif self.marvel or self.trippy:
            base_color = self.sdf.get_textures(points)
        else:
            base_color = 1.0

        return base_color * self.feature * points.new_ones(points.shape[0], 1)
    
    def forward(self, points):
        return self.get_distance(points)



class HarmonicEmbedding(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        n_harmonic_functions: int = 6,
        omega0: float = 1.0,
        logspace: bool = True,
        include_input: bool = True,
    ) -> None:
        super().__init__()

        if logspace:
            frequencies = 2.0 ** torch.arange(
                n_harmonic_functions,
                dtype=torch.float32,
            )
        else:
            frequencies = torch.linspace(
                1.0,
                2.0 ** (n_harmonic_functions - 1),
                n_harmonic_functions,
                dtype=torch.float32,
            )

        self.register_buffer("_frequencies", omega0 * frequencies, persistent=False)
        self.include_input = include_input
        self.output_dim = n_harmonic_functions * 2 * in_channels

        if self.include_input:
            self.output_dim += in_channels

    def forward(self, x: torch.Tensor):
        embed = (x[..., None] * self._frequencies).view(*x.shape[:-1], -1)

        if self.include_input:
            return torch.cat((embed.sin(), embed.cos(), x), dim=-1)
        else:
            return torch.cat((embed.sin(), embed.cos()), dim=-1)


class LinearWithRepeat(torch.nn.Linear):
    def forward(self, input):
        n1 = input[0].shape[-1]
        output1 = F.linear(input[0], self.weight[:, :n1], self.bias)
        output2 = F.linear(input[1], self.weight[:, n1:], None)
        return output1 + output2.unsqueeze(-2)


class MLPWithInputSkips(torch.nn.Module):
    def __init__(
        self,
        n_layers: int,
        input_dim: int,
        output_dim: int,
        skip_dim: int,
        hidden_dim: int,
        input_skips,
    ):
        super().__init__()

        layers = []

        for layeri in range(n_layers):
            if layeri == 0:
                dimin = input_dim
                dimout = hidden_dim
            elif layeri in input_skips:
                dimin = hidden_dim + skip_dim
                dimout = hidden_dim
            else:
                dimin = hidden_dim
                dimout = hidden_dim

            linear = torch.nn.Linear(dimin, dimout)
            layers.append(torch.nn.Sequential(linear, torch.nn.ReLU(True)))

        self.mlp = torch.nn.ModuleList(layers)
        self._input_skips = set(input_skips)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        y = x

        for li, layer in enumerate(self.mlp):
            if li in self._input_skips:
                y = torch.cat((y, z), dim=-1)

            y = layer(y)

        return y


class NeuralSurface(torch.nn.Module):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()

        # TODO (Q2): Implement Neural Surface MLP to output per-point SDF
        # TODO (Q3): Implement Neural Surface MLP to output per-point color

        self.harmonic_embedding_xyz = HarmonicEmbedding(3,
                                                        cfg.n_harmonic_functions_xyz)
        embedding_dim_xyz = self.harmonic_embedding_xyz.output_dim
        n_hidden_dist = cfg.n_hidden_neurons_distance
        n_layers_dist = cfg.n_layers_distance
        n_hidden_color = cfg.n_hidden_neurons_color
        n_layers_color = cfg.n_layers_color

        mlp_common = []
        mlp_color = []
        mlp_dist = []
        input_size = embedding_dim_xyz

        for i in range(n_layers_dist):
            mlp_common.append(torch.nn.Linear(input_size, n_hidden_dist))
            mlp_common.append(torch.nn.ReLU())
            input_size = n_hidden_dist

        # Distance can be anything. Hence, no relu or sigmoid.
        mlp_dist.append(torch.nn.Linear(input_size, 1))
        
        for i in range(n_layers_color):
            mlp_color.append(torch.nn.Linear(input_size, n_hidden_color))
            mlp_color.append(torch.nn.ReLU())
            input_size = n_hidden_color

        # Range for each would be between 0 and 1.
        mlp_color.append(torch.nn.Linear(input_size, 3))
        mlp_color.append(torch.nn.Sigmoid())

        self.mlp_common = torch.nn.Sequential(*mlp_common)
        self.mlp_dist = torch.nn.Sequential(*mlp_dist)
        self.mlp_color = torch.nn.Sequential(*mlp_color)

    def get_distance(
        self,
        points
    ):
        '''
        TODO: Q2
        Output:
            distance: N X 1 Tensor, where N is number of input points
        '''
        points = points.view(-1, 3)
        
        embed_points = self.harmonic_embedding_xyz(points)
        feat = self.mlp_common(embed_points)
        dists = self.mlp_dist(feat)

        return dists
    
    def get_color(
        self,
        points
    ):
        '''
        TODO: Q3
        Output:
            colors: N X 3 Tensor, where N is number of input points
        '''
        points = points.view(-1, 3)
        
        embed_points = self.harmonic_embedding_xyz(points)
        feat = self.mlp_common(embed_points)
        colors = self.mlp_color(feat)
        
        return colors

    def get_distance_color(
        self,
        points
    ):
        '''
        TODO: Q3
        Output:
            distance, colors: N X 1, N X 3 Tensors, where N is number of input points
        You may just implement this by independent calls to get_distance, get_color
            but, depending on your MLP implementation, it maybe more efficient to share some computation
        '''

        points = points.view(-1, 3)
        
        embed_points = self.harmonic_embedding_xyz(points)
        feat = self.mlp_common(embed_points)

        dists = self.mlp_dist(feat)
        colors = self.mlp_color(feat)
        
        return dists, colors
        
    def forward(self, points):
        return self.get_distance(points)

    def get_distance_and_gradient(
        self,
        points
    ):
        has_grad = torch.is_grad_enabled()
        points = points.view(-1, 3)

        # Calculate gradient with respect to points
        with torch.enable_grad():
            points = points.requires_grad_(True)
            distance = self.get_distance(points)
            gradient = autograd.grad(
                distance,
                points,
                torch.ones_like(distance, device=points.device),
                create_graph=has_grad,
                retain_graph=has_grad,
                only_inputs=True
            )[0]
        
        # print('dists', distance.shape)

        return distance, gradient


implicit_dict = {
    'sdf_surface': SDFSurface,
    'neural_surface': NeuralSurface,
}
