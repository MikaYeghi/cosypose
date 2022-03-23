from tracemalloc import start
from cosypose.datasets.datasets_cfg import make_ply_dataset
from cosypose.lib3d.transform_ops import invert_T
from cosypose.utils.logging import get_logger

from pytorch3d.io import load_ply
from pytorch3d.structures import Meshes, join_meshes_as_batch
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
import os
from cosypose.features.feature_loader import FeatureLoader
from cosypose.rendering.feature_shader import FeatureShader
from tqdm import tqdm

import pdb
from pprint import pprint
import time

logger = get_logger(__name__)

class Pytorch3DSceneRenderer(torch.nn.Module):
    def __init__(self,
                save_dir,
                ply_ds='ycbv',
                device='cpu',
                n_feature_channels=64,
                features_on=False,
                features_dict=None):
        super(Pytorch3DSceneRenderer, self).__init__()
        self.ply_ds = make_ply_dataset(ply_ds)
        self.opencv_to_pytorch3d = torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=torch.float, device=device) # conversion matrix from opencv to pytorch3d
        # Connect to the device
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
            torch.cuda.set_device(self.device)
            logger.info("Using CUDA device for rendering")
        else:
            self.device = torch.device("cpu")
            logger.info("Using CPU for rendering")

        # Load the dataset of CAD models
        self.objects = self.load_dataset()

        # Load the feature loader
        self.features_on = features_on
        if features_on:
            self.feature_loader = FeatureLoader(save_dir=save_dir, number_of_channels=n_feature_channels, features_dict=features_dict) # loads features for CAD models
            self.n_feature_channels = n_feature_channels
        else:
            self.n_feature_channels = 3

    def render(self, obj_infos, TCO, K, resolution=(240, 320)):
        TCO = torch.as_tensor(TCO).clone().detach()
        K = torch.as_tensor(K)
        bsz = len(TCO)
        assert TCO.shape == (bsz, 4, 4)
        assert K.shape == (bsz, 3, 3)

        rendered_images = torch.empty((1, resolution[0], resolution[1], self.n_feature_channels)).to(self.device)

        object_meshes = self.load_objects_batch(obj_infos, bsz, device=self.device)
        renderer = self.setup_batch_scenes(resolution, K, TCO, bsz, device=self.device)
        images = self.shot_batch(object_meshes, renderer)

        rendered_images = images.permute(0, 3, 1, 2)
        return rendered_images

    def load_object(self, obj_info, device='cpu'):
        label = obj_info['name']
        verts, faces = self.objects[label]
        verts.to(device)
        faces.to(device)

        # Generate the mesh
        if self.features_on:
            verts_features = self.feature_loader.get_features()[label]
        else:
            colour = 1.0
            verts_features=torch.tensor([[colour, colour, colour] for _ in range(len(verts))])
        verts_features = torch.unsqueeze(verts_features, 0).to(device)
        textures = TexturesVertex(verts_features=verts_features)
        mesh = Meshes(verts=[verts], faces=[faces], textures=textures).to(device)
        
        return mesh

    def load_objects_batch(self, obj_infos, bsz, device='cpu'):
        meshes_list = list()
        for n in np.arange(bsz):
            obj_info = dict(
                name=obj_infos[n]['name'],
                TWO=np.eye(4)
            )
            object_mesh = self.load_object(obj_info, device=self.device)
            meshes_list.append(object_mesh)
        mesh = join_meshes_as_batch(meshes_list).to(device)
        return mesh

    def load_dataset(self):
        """
        This function loads all .ply files from the ply dataset path into a dictionary.
        """
        assert self.ply_ds is not None
        obj_ds = dict()

        logger.info("Loading objects into memory...")
        for obj_ in tqdm(self.ply_ds):
            label = obj_.label
            obj_path, scale = self.ply_ds.get_ply_path_by_label(label)
            verts, faces = load_ply(obj_path)

            # Scale the object as in the original paper
            verts = verts * scale # scale vertices

            # Save in the dictionary
            obj_ds[label] = (verts, faces)

        return obj_ds

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
            lights = DirectionalLights(device=self.device, direction=((0,0,-1),)).to(self.device)
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

    def setup_batch_scenes(self, resolution, K, TWC, bsz, device='cpu'):
        R_list = list()     # List of rotation matrices
        T_list = list()     # List of translation vectors
        FL_list = list()    # List of focal lengths
        PP_list = list()    # List of principal points

        for n in np.arange(bsz):
            K_ = K[n]
            TWC_ = TWC[n]

            # Extract camera intrinsic parameters
            fx = K_[0][0]
            fy = K_[1][1]
            sx = K_[0][2]
            sy = K_[1][2]
            focal_length = torch.tensor([fx, fy]).to(self.device)
            principal_point = torch.tensor([sx, sy]).to(self.device)

            # Extract rotation and translation matrices
            R = TWC_[:3, :3].to(self.device)
            T = TWC_[:3, -1].to(self.device)

            # Convert from opencv to pytorch3d coordinates
            R = R.T
            R = torch.matmul(R, self.opencv_to_pytorch3d)
            T[0] = -1 * T[0]
            T[1] = -1 * T[1]

            # Append to the lists of batches
            R_list.append(R.unsqueeze(0))
            T_list.append(T.unsqueeze(0))
            FL_list.append(focal_length.unsqueeze(0))
            PP_list.append(principal_point.unsqueeze(0))
        
        R = torch.cat(R_list).to(device)
        T = torch.cat(T_list).to(device)
        FL = torch.cat(FL_list).to(device)
        PP = torch.cat(PP_list).to(device)

        # Set up camera, object and world
        cameras = PerspectiveCameras(device=self.device, 
                                R=R, 
                                T=T, 
                                focal_length=FL, 
                                principal_point=PP,
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
            lights = DirectionalLights(device=self.device, direction=((0,0,-1),)).to(self.device)
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
        Returns an image of shape (H, W, C).
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
        # plt.imshow(image[0][0].cpu().numpy())
        # plt.show()

        return image

    def shot_batch(self, object_meshes, renderer):
        """
        Returns a batch of images.
        """
        images = renderer(object_meshes)

        if self.features_on: # features case
            images = images.permute(3, 0, 1, 2, 4)
            images = images[0]
        else: # RGB case
            images = images[..., :3] # pick RGB channels

            """
            Use the lines below to plot rendered example images.
            """
            # for image in images:
            #     from matplotlib import pyplot as plt
            #     plt.imshow(image.cpu().detach().numpy())
            #     plt.show()

        return images

    def save_features_dict(self, save_dir, verbose=1):
        assert self.features_on
        self.feature_loader.save_features(save_dir=save_dir, verbose=verbose)