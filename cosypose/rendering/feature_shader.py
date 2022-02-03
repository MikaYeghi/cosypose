# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from typing import Optional

import torch
import torch.nn as nn

from pytorch3d.common.types import Device
from pytorch3d.structures.meshes import Meshes
from pytorch3d.renderer.blending import (
    BlendParams,
    hard_rgb_blend,
    sigmoid_alpha_blend,
    softmax_rgb_blend,
)
from pytorch3d.renderer.lighting import PointLights
from pytorch3d.renderer.materials import Materials
from pytorch3d.renderer.utils import TensorProperties
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.renderer.mesh.shading import flat_shading, gouraud_shading, phong_shading
from pytorch3d.ops import interpolate_face_attributes
from typing import Union

import pdb

def multichannel_shading(
    meshes, fragments, cameras, texels
) -> torch.Tensor:
    """
    Apply per pixel shading. First interpolate the vertex normals and
    vertex coordinates using the barycentric coordinates to get the position
    and normal at each pixel. Then compute the illumination for each pixel.
    The pixel color is obtained by multiplying the pixel textures by the ambient
    and diffuse illumination and adding the specular component.

    Args:
        meshes: Batch of meshes
        fragments: Fragments named tuple with the outputs of rasterization
        lights: Lights class containing a batch of lights
        cameras: Cameras class containing a batch of cameras
        materials: Materials class containing a batch of material properties
        texels: texture per pixel of shape (N, H, W, K, 3)

    Returns:
        colors: (N, H, W, K, 3)
    """
    verts = meshes.verts_packed()  # (V, 3)
    faces = meshes.faces_packed()  # (F, 3)
    vertex_normals = meshes.verts_normals_packed()  # (V, 3)
    faces_verts = verts[faces]
    faces_normals = vertex_normals[faces]
    pixel_coords = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, faces_verts
    )
    pixel_normals = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, faces_normals
    )
    # ambient, diffuse, specular = _apply_lighting(
    #     pixel_coords, pixel_normals, lights, cameras, materials
    # )
    # colors = (ambient + diffuse) * texels + specular
    colors = texels
    return colors

def softmax_multichannel_blend(
    colors: torch.Tensor,
    fragments,
    blend_params: BlendParams,
    znear: Union[float, torch.Tensor] = 1.0,
    zfar: Union[float, torch.Tensor] = 100,
    n_channels=3
) -> torch.Tensor:
    """
    RGB and alpha channel blending to return an RGBA image based on the method
    proposed in [1]
      - **RGB** - blend the colors based on the 2D distance based probability map and
        relative z distances.
      - **A** - blend based on the 2D distance based probability map.

    Args:
        colors: (N, H, W, K, 3) RGB color for each of the top K faces per pixel.
        fragments: namedtuple with outputs of rasterization. We use properties
            - pix_to_face: LongTensor of shape (N, H, W, K) specifying the indices
              of the faces (in the packed representation) which
              overlap each pixel in the image.
            - dists: FloatTensor of shape (N, H, W, K) specifying
              the 2D euclidean distance from the center of each pixel
              to each of the top K overlapping faces.
            - zbuf: FloatTensor of shape (N, H, W, K) specifying
              the interpolated depth from each pixel to to each of the
              top K overlapping faces.
        blend_params: instance of BlendParams dataclass containing properties
            - sigma: float, parameter which controls the width of the sigmoid
              function used to calculate the 2D distance based probability.
              Sigma controls the sharpness of the edges of the shape.
            - gamma: float, parameter which controls the scaling of the
              exponential function used to control the opacity of the color.
            - background_color: (3) element list/tuple/torch.Tensor specifying
              the RGB values for the background color.
        znear: float, near clipping plane in the z direction
        zfar: float, far clipping plane in the z direction

    Returns:
        RGBA pixel_colors: (N, H, W, 4)

    [0] Shichen Liu et al, 'Soft Rasterizer: A Differentiable Renderer for
    Image-based 3D Reasoning'
    """

    N, H, W, K = fragments.pix_to_face.shape
    device = fragments.pix_to_face.device
    pixel_colors = torch.ones((N, H, W, n_channels), dtype=colors.dtype, device=colors.device)
    # background_ = blend_params.background_color
    background_ = torch.ones((n_channels), dtype=colors.dtype, device=colors.device)
    if not isinstance(background_, torch.Tensor):
        background = torch.tensor(background_, dtype=torch.float32, device=device)
    else:
        background = background_.to(device)

    # Weight for background color
    eps = 1e-10

    # Mask for padded pixels.
    mask = fragments.pix_to_face >= 0

    # Sigmoid probability map based on the distance of the pixel to the face.
    prob_map = torch.sigmoid(-fragments.dists / blend_params.sigma) * mask

    # The cumulative product ensures that alpha will be 0.0 if at least 1
    # face fully covers the pixel as for that face, prob will be 1.0.
    # This results in a multiplication by 0.0 because of the (1.0 - prob)
    # term. Therefore 1.0 - alpha will be 1.0.
    alpha = torch.prod((1.0 - prob_map), dim=-1)

    # Weights for each face. Adjust the exponential by the max z to prevent
    # overflow. zbuf shape (N, H, W, K), find max over K.
    # TODO: there may still be some instability in the exponent calculation.

    # Reshape to be compatible with (N, H, W, K) values in fragments
    if torch.is_tensor(zfar):
        # pyre-fixme[16]
        zfar = zfar[:, None, None, None]
    if torch.is_tensor(znear):
        znear = znear[:, None, None, None]

    z_inv = (zfar - fragments.zbuf) / (zfar - znear) * mask
    # pyre-fixme[16]: `Tuple` has no attribute `values`.
    z_inv_max = torch.max(z_inv, dim=-1).values[..., None].clamp(min=eps)
    weights_num = prob_map * torch.exp((z_inv - z_inv_max) / blend_params.gamma)

    # Also apply exp normalize trick for the background color weight.
    # Clamp to ensure delta is never 0.
    # pyre-fixme[6]: Expected `Tensor` for 1st param but got `float`.
    delta = torch.exp((eps - z_inv_max) / blend_params.gamma).clamp(min=eps)

    # Normalize weights.
    # weights_num shape: (N, H, W, K). Sum over K and divide through by the sum.
    denom = weights_num.sum(dim=-1)[..., None] + delta

    # Sum: weights * textures + background color
    weighted_colors = (weights_num[..., None] * colors).sum(dim=-2)
    weighted_background = delta * background
    pixel_colors = (weighted_colors + weighted_background) / denom
    pixel_colors = 1.0 - alpha

    return pixel_colors

class FeatureShader(nn.Module):
    """
    Per pixel lighting - the lighting model is applied using the interpolated
    coordinates and normals for each pixel. The blending function returns the
    soft aggregated color using all the faces per pixel.

    To use the default values, simply initialize the shader with the desired
    device e.g.

    .. code-block::

        shader = SoftPhongShader(device=torch.device("cuda:0"))
    """

    def __init__(
        self,
        device: Device = "cpu",
        cameras: Optional[TensorProperties] = None,
        # lights: Optional[TensorProperties] = None,
        # materials: Optional[Materials] = None,
        blend_params: Optional[BlendParams] = None,
        n_channels=32
    ) -> None:
        super().__init__()
        # self.lights = lights if lights is not None else PointLights(device=device)
        # self.materials = (
        #     materials if materials is not None else Materials(device=device)
        # )
        self.cameras = cameras
        self.blend_params = blend_params if blend_params is not None else BlendParams()
        self.n_channels = n_channels

    def to(self, device: Device):
        # Manually move to device modules which are not subclasses of nn.Module
        cameras = self.cameras
        if cameras is not None:
            self.cameras = cameras.to(device)
        # self.materials = self.materials.to(device)
        # self.lights = self.lights.to(device)
        return self

    def forward(self, fragments: Fragments, meshes: Meshes, **kwargs) -> torch.Tensor:
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of SoftPhongShader"
            raise ValueError(msg)

        texels = meshes.sample_textures(fragments)
        # lights = kwargs.get("lights", self.lights)
        # materials = kwargs.get("materials", self.materials)
        # blend_params = kwargs.get("blend_params", self.blend_params)
        # colors = multichannel_shading(
        #     meshes=meshes,
        #     fragments=fragments,
        #     texels=texels,
        #     cameras=cameras
        # )
        # znear = kwargs.get("znear", getattr(cameras, "znear", 1.0))
        # zfar = kwargs.get("zfar", getattr(cameras, "zfar", 100.0))
        # images = softmax_multichannel_blend(
        #     colors, fragments, blend_params, znear=znear, zfar=zfar, n_channels=self.n_channels
        # )
        images = texels

        return images