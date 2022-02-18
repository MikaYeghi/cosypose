import torch
from torch import nn

from cosypose.config import DEBUG_DATA_DIR
from cosypose.lib3d.camera_geometry import get_K_crop_resize, boxes_from_uv

from cosypose.lib3d.cropping import deepim_crops_robust as deepim_crops
from cosypose.lib3d.camera_geometry import project_points_robust as project_points

from cosypose.lib3d.rotations import (
    compute_rotation_matrix_from_ortho6d, compute_rotation_matrix_from_quaternions)
from cosypose.lib3d.cosypose_ops import apply_imagespace_predictions

from cosypose.utils.logging import get_logger
logger = get_logger(__name__)

import numpy as np
import pdb
from cosypose.models.embednet import embednet
import cv2
from tqdm import tqdm
from matplotlib import pyplot as plt


class PosePredictor(nn.Module):
    def __init__(self, backbone, renderer,
                 mesh_db, render_size=(240, 320),
                 pose_dim=9):
        super().__init__()

        self.backbone = backbone
        self.renderer = renderer
        self.mesh_db = mesh_db
        self.render_size = render_size
        self.pose_dim = pose_dim
        try:
            self.features_on = self.renderer.features_on # [MIKAEL] if true, then uses features instead of RGB
        except Exception:
            self.features_on = False # set to false if the renderer has no features_no (means that it's not pytorch3d)

        n_features = backbone.n_features

        self.heads = dict()
        self.pose_fc = nn.Linear(n_features, pose_dim, bias=True)
        self.heads['pose'] = self.pose_fc

        self.debug = False
        self.tmp_debug = dict()

        # Initialize the Embeddings Network
        if self.features_on:
            self.embednet = embednet(backbone='resnet34', pretrained=True).cuda()

    def enable_debug(self):
        self.debug = True

    def disable_debug(self):
        self.debug = False

    def crop_inputs(self, images, K, TCO, labels):
        bsz, nchannels, h, w = images.shape
        assert K.shape == (bsz, 3, 3)
        assert TCO.shape == (bsz, 4, 4)
        assert len(labels) == bsz
        meshes = self.mesh_db.select(labels)
        points = meshes.sample_points(2000, deterministic=True)
        uv = project_points(points, K, TCO)
        boxes_rend = boxes_from_uv(uv)
        boxes_crop, images_cropped = deepim_crops(
            images=images, obs_boxes=boxes_rend, K=K,
            TCO_pred=TCO, O_vertices=points, output_size=self.render_size, lamb=1.4
        )
        K_crop = get_K_crop_resize(K=K.clone(), boxes=boxes_crop,
                                   orig_size=images.shape[-2:], crop_resize=self.render_size)
        if self.debug:
            self.tmp_debug.update(
                boxes_rend=boxes_rend,
                rend_center_uv=project_points(torch.zeros(bsz, 1, 3).to(K.device), K, TCO),
                uv=uv,
                boxes_crop=boxes_crop,
            )
        return images_cropped, K_crop.detach(), boxes_rend, boxes_crop

    def update_pose(self, TCO, K_crop, pose_outputs):
        if self.pose_dim == 9:
            dR = compute_rotation_matrix_from_ortho6d(pose_outputs[:, 0:6])
            vxvyvz = pose_outputs[:, 6:9]
        elif self.pose_dim == 7:
            dR = compute_rotation_matrix_from_quaternions(pose_outputs[:, 0:4])
            vxvyvz = pose_outputs[:, 4:7]
        else:
            raise ValueError(f'pose_dim={self.pose_dim} not supported')
        TCO_updated = apply_imagespace_predictions(TCO, K_crop, vxvyvz, dR)
        return TCO_updated

    def net_forward(self, x):
        x = self.backbone(x)
        x = x.flatten(2).mean(dim=-1)
        outputs = dict()
        for k, head in self.heads.items():
            outputs[k] = head(x)
        return outputs

    def plot_crop(self, image, K, boxes_crop, label):
        """
        This function plots cropped images.
        """
        from matplotlib import pyplot as plt
        print(label)
        print(boxes_crop)
        print(image.shape)
        crop_width = int(boxes_crop[2] - boxes_crop[0])
        crop_height = int(boxes_crop[3] - boxes_crop[1])
        print(crop_width)
        print(crop_height)
        plt.imshow(image.cpu())
        plt.show()
        height, width = image.shape[:2]
        final_image = image.clone().detach()
        for i in tqdm(range(height)):
            for j in range(width):
                old_vector = torch.tensor([j, i, 1.0], dtype=torch.float32, device='cuda:0')
                new_vector = torch.matmul(K, old_vector).to('cuda:0')
                new_vector = new_vector / new_vector[2]
                new_vector = torch.round(new_vector).int()[:2]

                # if 0 <= new_vector[1] < crop_height and 0 <= new_vector[0] < crop_width:
                if 0 <= new_vector[1] < 240 and 0 <= new_vector[0] < 320:
                    # final_image[i, j] = image[new_vector[1], new_vector[0]]
                    final_image[i, j] = image[i, j]
                else:
                    final_image[i, j] = 0.0
        plt.imshow(final_image.cpu())
        plt.show()
        pdb.set_trace()

    def forward(self, images, K, labels, TCO, n_iterations=1):
        bsz, nchannels, h, w = images.shape
        assert K.shape == (bsz, 3, 3)
        assert TCO.shape == (bsz, 4, 4)
        assert len(labels) == bsz

        outputs = dict()
        TCO_input = TCO
        for n in range(n_iterations):
            TCO_input = TCO_input.detach()
            images_crop, K_crop, boxes_rend, boxes_crop = self.crop_inputs(images, K, TCO_input, labels)
            # K_new = torch.matmul(K_crop[0], torch.inverse(K[0]))
            # image = images[0]
            # for image in images_crop:
            #     image = image.permute(1,2,0)
            #     plt.imshow(image.cpu().numpy())
            #     plt.show()
            # self.plot_crop(image, K_new, boxes_crop[0], labels[0])

            # print("ORIGINAL K, CROP K AND BOXES CROP")
            # print(K)
            # print(K_crop)
            # print(boxes_crop)
            # print(TCO_input)
            # print("RENDER BELOW")
            if self.features_on:
                images_crop = self.embednet(images_crop)
                K_crop = get_K_crop_resize(K=K.clone(), boxes=boxes_crop,
                                   orig_size=images.shape[-2:], crop_resize=images_crop.shape[-2:])

            renders = self.renderer.render(obj_infos=[dict(name=l) for l in labels],
                                           TCO=TCO_input,
                                           K=K_crop, resolution=images_crop.shape[-2:])

            x = torch.cat((images_crop, renders), dim=1)

            model_outputs = self.net_forward(x)

            TCO_output = self.update_pose(TCO_input, K_crop, model_outputs['pose'])

            outputs[f'iteration={n+1}'] = {
                'TCO_input': TCO_input,
                'TCO_output': TCO_output,
                'K_crop': K_crop,
                'model_outputs': model_outputs,
                'boxes_rend': boxes_rend,
                'boxes_crop': boxes_crop,
            }

            TCO_input = TCO_output

            if self.debug:
                self.tmp_debug.update(outputs[f'iteration={n+1}'])
                self.tmp_debug.update(
                    images=images,
                    images_crop=images_crop,
                    renders=renders,
                )
                path = DEBUG_DATA_DIR / f'debug_iter={n+1}.pth.tar'
                logger.info(f'Wrote debug data: {path}')
                torch.save(self.tmp_debug, path)

        return outputs
