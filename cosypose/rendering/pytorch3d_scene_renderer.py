from tracemalloc import start
from cosypose.datasets.datasets_cfg import make_ply_dataset
from cosypose.lib3d.transform_ops import invert_T

from pytorch3d.io import load_ply
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    DirectionalLights,
    AmbientLights,
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,
    HardPhongShader,
    SoftPhongShader,
    TexturesVertex,
    TexturesAtlas,
    HardFlatShader
)
from pytorch3d.renderer.blending import BlendParams

import torch
import numpy as np
from cosypose.features.feature_loader import FeatureLoader
from cosypose.rendering.feature_shader import FeatureShader

import pdb
from pprint import pprint
import time

class Pytorch3DSceneRenderer:
    def __init__(self,
                ply_ds='ycbv',
                device='cpu',
                n_feature_channels=64,
                features_on=False):
        self.ply_ds = make_ply_dataset(ply_ds)
        self.opencv_to_pytorch3d = torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=torch.float, device=device) # conversion matrix from opencv to pytorch3d
        # Connect to the device
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
            torch.cuda.set_device(self.device)
            print("Using CUDA device for rendering")
        else:
            self.device = torch.device("cpu")
            print("Using CPU for rendering")

        # Load the feature loader
        self.features_on = features_on
        if features_on:
            self.feature_loader = FeatureLoader(number_of_channels=n_feature_channels) # [MIKAEL] loads features for CAD models
            self.n_feature_channels = n_feature_channels
        else:
            self.n_feature_channels = 3

    def render(self, obj_infos, TCO, K, resolution=(240, 320)):
        TCO = torch.as_tensor(TCO).detach()
        # TOC = invert_T(TCO).cpu().numpy()
        # TOC = TCO.detach().cpu().numpy()
        # K = torch.as_tensor(K).cpu().numpy()
        K = torch.as_tensor(K)
        bsz = len(TCO)
        assert TCO.shape == (bsz, 4, 4)
        assert K.shape == (bsz, 3, 3)

        rendered_images = torch.empty((1, resolution[0], resolution[1], self.n_feature_channels)).to(self.device)

        for n in np.arange(bsz):
            obj_info = dict(
                name=obj_infos[n]['name'],
                TWO=np.eye(4)
            )
            cam_info = dict(
                resolution=resolution,
                K=K[n],
                # TWC=TOC[n],
                TWC=TCO[n],
            )

            object_mesh = self.load_object(obj_info) # load the object mesh
            renderer = self.setup_scene(cam_info) # setup the scene
            image = self.shot(object_mesh, renderer, resolution) # render the image

            rendered_images = torch.cat((rendered_images, image), 0)

        rendered_images = rendered_images[1:] # takes care of the first empty tensor
        rendered_images = rendered_images.permute(0, 3, 1, 2)
        return rendered_images


    def load_object(self, obj_info):
        label = obj_info['name']
        obj_path, scale = self.ply_ds.get_ply_path_by_label(label)
        verts, faces = load_ply(obj_path)

        # Scale the object as in the original paper
        # verts = torch.tensor([[x[0] * scale, x[1] * scale, x[2] * scale] for x in verts], dtype=verts.dtype)
        verts.to(self.device)
        faces.to(self.device)
        verts = verts * scale # scale vertices

        # Generate the mesh
        if self.features_on:
            verts_features = self.feature_loader.get_features()[label]
        else:
            colour = 1.35
            verts_features=torch.tensor([[colour, colour, colour] for _ in range(len(verts))])
        verts_features = torch.unsqueeze(verts_features, 0).to(self.device)
        textures = TexturesVertex(verts_features=verts_features)
        mesh = Meshes(verts=[verts], faces=[faces], textures=textures).to(self.device)
        
        return mesh

    def setup_scene(self, cam_info):
        K = cam_info['K']
        # TWC = torch.from_numpy(cam_info['TWC'])
        TWC = cam_info['TWC']
        resolution = cam_info['resolution']
        
        # Extract camera intrinsic parameters
        fx = K[0][0]
        fy = K[1][1]
        sx = K[0][2]
        sy = K[1][2]
        focal_length = torch.tensor([fx, fy]).to(self.device)
        principal_point = torch.tensor([sx, sy]).to(self.device)

        # Extract rotation and translation matrices
        R = TWC[:3, :3].to(self.device)
        T = TWC[:3, -1].to(self.device)

        # Convert from opencv to pytorch3d coordinates
        R = R.T
        R = torch.matmul(R, self.opencv_to_pytorch3d)
        T[0] = -1 * T[0]
        T[1] = -1 * T[1]

        # Set up camera, object and world
        cameras = PerspectiveCameras(device=self.device, 
                                R=R.unsqueeze(0), 
                                T=T.unsqueeze(0), 
                                focal_length=focal_length.unsqueeze(0), 
                                principal_point=principal_point.unsqueeze(0),
                                in_ndc=False,
                                image_size=[resolution])

        raster_settings = RasterizationSettings(image_size=resolution,
                                                blur_radius=0.0, 
                                                faces_per_pixel=1,
                                                max_faces_per_bin=150000,
                                            )

        if self.features_on:
            shader = FeatureShader(
                device=self.device, 
                cameras=cameras,
            )
        else:
            lights = DirectionalLights(device=self.device, direction=((0,0,1),)).to(self.device)
            shader = SoftPhongShader(
                device=self.device, 
                cameras=cameras,
                lights=lights,
                blend_params=BlendParams(background_color=(0.0, 0.0, 0.0))
            )

        rasterizer = MeshRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings
            )

        renderer = MeshRenderer(
            rasterizer=rasterizer,
            shader=shader
        ).to(self.device)

        return renderer


    def shot(self, object_mesh, renderer, resolution):
        """
        Returns an image of shape (h, w, 3) and type numpy array.
        """
        image = renderer(object_mesh)
        image = image[0, ..., :self.n_feature_channels]

        if self.features_on:
            image = image.permute(2, 0, 1, 3)
        else:
            image = image.unsqueeze(0)
        
        """
        Use the lines below to plot rendered images (only if using features_on=False).
        """
        # from matplotlib import pyplot as plt
        # plt.imshow(image[0].cpu().numpy())
        # plt.show()

        return image


