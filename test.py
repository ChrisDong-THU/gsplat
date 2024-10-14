import torch

from gsplat.project_gaussians_2d_scale_rot import project_gaussians_2d_scale_rot
from gsplat.rasterize_sum import rasterize_gaussians_sum

import matplotlib.pyplot as plt

device = torch.device("cuda:0")
num_points = 1000

# _xy = 2 * (torch.rand(num_points, 2) - 0.5)
# _scale_bound = torch.tensor([1, 1]).view(1, 2)
# _scaling = torch.abs(torch.rand(num_points, 2) + _scale_bound)
# _rotation = (torch.rand(num_points, 1)*2*torch.pi)
# _opacity = torch.ones((num_points, 1))
# _feature = torch.ones((num_points, 3))

_xy = torch.tensor([[0.0, 0.0]]).tile(num_points, 1)
_scaling = torch.tensor([[1.0, 1.0]]).tile(num_points, 1)
_rotation = torch.tensor([[0.0]]).tile(num_points, 1)
_opacity = torch.tensor([[1.0]]).tile(num_points, 1)
_feature = torch.tensor([[1.0, 1.0, 1.0]]).tile(num_points, 1)

_xy = _xy.to(device)
_scaling = _scaling.to(device)
_rotation = _rotation.to(device)
_opacity = _opacity.to(device)
_feature = _feature.to(device)

background = torch.ones(3, device=device)

img_H = 256
img_W = 256
block_H = 16
block_W = 16
tile_bounds = (
    (img_W + block_W - 1) // block_W,
    (img_H + block_H - 1) // block_H,
    1,
) # 3个维度上分块数量

xys, depths, radii, conics, num_tiles_hit = project_gaussians_2d_scale_rot(_xy, _scaling, _rotation, img_H, img_W, tile_bounds)
out_img = rasterize_gaussians_sum(xys, depths, radii, conics, num_tiles_hit, _feature, _opacity, img_H, img_W, block_H, block_W, background=background, return_alpha=False)

out_img = out_img.clamp(0, 1)

out_img_np = out_img.cpu().numpy()

plt.imshow(out_img_np)
plt.show()

pass