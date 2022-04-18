import torch
import numpy as np

from cosypose.lib3d.cosypose_ops import TCO_init_from_boxes, TCO_init_from_boxes_zup_autodepth
from cosypose.lib3d.transform_ops import add_noise
from cosypose.lib3d.cosypose_ops import (
    loss_refiner_CO_disentangled,
    loss_refiner_CO_disentangled_quaternions,
)
from cosypose.lib3d.mesh_losses import compute_ADD_L1_loss
from torch.nn import MSELoss

import pdb
from matplotlib import pyplot as plt


def cast(obj):
    return obj.cuda(non_blocking=True)


def feature_loss(feature_maps, renders, device='cpu'):
    loss_ = MSELoss()
    return loss_(feature_maps.detach().to(device), renders.to(device))


def h_pose(model, mesh_db, data, meters,
           cfg, n_iterations=1, input_generator='fixed'):

    batch_size, _, h, w = data.images.shape

    images = cast(data.images).float() / 255.
    K = cast(data.K).float()
    TCO_gt = cast(data.TCO).float()
    labels = np.array([obj['name'] for obj in data.objects])
    bboxes = cast(data.bboxes).float()

    meshes = mesh_db.select(labels)
    points = meshes.sample_points(cfg.n_points_loss, deterministic=False)
    TCO_possible_gt = TCO_gt.unsqueeze(1) @ meshes.symmetries

    if input_generator == 'fixed':
        TCO_init = TCO_init_from_boxes(z_range=(1.0, 1.0), boxes=bboxes, K=K)
    elif input_generator == 'gt+noise':
        TCO_init = add_noise(TCO_possible_gt[:, 0], euler_deg_std=[15, 15, 15], trans_std=[0.01, 0.01, 0.05])
    elif input_generator == 'fixed+trans_noise':
        assert cfg.init_method == 'z-up+auto-depth'
        TCO_init = TCO_init_from_boxes_zup_autodepth(bboxes, points, K)
        TCO_init = add_noise(TCO_init,
                             euler_deg_std=[0, 0, 0],
                             trans_std=[0.01, 0.01, 0.05])
    else:
        raise ValueError('Unknown input generator', input_generator)

    # model.module.enable_debug()
    TCO_init = TCO_gt.detach().clone()
    outputs = model(images=images, K=K, labels=labels,
                    TCO=TCO_init, n_iterations=n_iterations)
    # raise ValueError

    losses_TCO_iter = []
    for n in range(n_iterations):
        iter_outputs = outputs[f'iteration={n+1}']
        K_crop = iter_outputs['K_crop']
        TCO_input = iter_outputs['TCO_input']
        TCO_pred = iter_outputs['TCO_output']
        model_outputs = iter_outputs['model_outputs']
        images_crop = iter_outputs['images_crop']
        renders = iter_outputs['renders']
        updated_renders = iter_outputs['updated_renders']

        if cfg.loss_disentangled:
            if cfg.n_pose_dims == 9:
                loss_fn = loss_refiner_CO_disentangled
            elif cfg.n_pose_dims == 7:
                loss_fn = loss_refiner_CO_disentangled_quaternions
            else:
                raise ValueError
                
            pose_outputs = model_outputs['pose']
            loss_TCO_iter = loss_fn(
                TCO_possible_gt=TCO_possible_gt,
                TCO_input=TCO_input,
                refiner_outputs=pose_outputs,
                K_crop=K_crop, points=points,
            )
            # Add loss function which penalizes difference between features
            feature_loss_ = feature_loss(images_crop, updated_renders, device='cuda:0')
            # print(f"Feature loss: {feature_loss_}")
            loss_TCO_iter += feature_loss_
        else:
            loss_TCO_iter = compute_ADD_L1_loss(
                TCO_possible_gt[:, 0], TCO_pred, points
            )

        meters[f'loss_TCO-iter={n+1}'].add(loss_TCO_iter.mean().item())
        losses_TCO_iter.append(loss_TCO_iter)

    loss_TCO = torch.cat(losses_TCO_iter).mean()
    loss = loss_TCO
    meters['loss_TCO'].add(loss_TCO.item())
    meters['loss_total'].add(loss.item())
    return loss
