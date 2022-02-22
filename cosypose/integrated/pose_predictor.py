import pyquaternion
import torch

from collections import defaultdict
from torch.utils.data import TensorDataset, DataLoader
from cosypose.lib3d.cosypose_ops import TCO_init_from_boxes, TCO_init_from_boxes_zup_autodepth

import cosypose.utils.tensor_collection as tc

from cosypose.utils.logging import get_logger
from cosypose.utils.timer import Timer
logger = get_logger(__name__)
from cosypose.datasets.samplers import DistributedSceneSampler
from cosypose.utils.distributed import get_world_size, get_rank
from cosypose.evaluation.data_utils import parse_obs_data
from cosypose.evaluation.eval_runner.gt_coarse_perturbation_eval import GroundTruthPerturbationEvaluationObject
from cosypose.integrated.multi_init import MultipleInitializer

import pdb
from tqdm import tqdm
import numpy as np
from math import sqrt, pi
from random import random


class CoarseRefinePosePredictor(torch.nn.Module):
    def __init__(self,
                 coarse_model=None,
                 refiner_model=None,
                 bsz_objects=64,
                 scene_ds=None,
                 use_gt=False):
        super().__init__()
        self.coarse_model = coarse_model
        self.refiner_model = refiner_model
        self.bsz_objects = bsz_objects
        self.use_gt = use_gt
        if self.use_gt:
            self.rank = get_rank()
            self.world_size = get_world_size()
            sampler = DistributedSceneSampler(scene_ds,
                                                num_replicas=self.world_size,
                                                rank=self.rank,
                                                shuffle=True)
            dataloader = DataLoader(scene_ds, batch_size=64,
                                    num_workers=4,
                                    sampler=sampler, collate_fn=self.collate_fn)
            self.ground_truth_data = list(tqdm(dataloader))
        self.scene_view_counter = {k+1: [] for k in range(20)} # used for counting which scenes and views have been processed
        self.eval()
    
    def collate_fn(self, batch):
        obj_data = []
        for data_n in batch:
            _, _, obs = data_n
            obj_data_ = parse_obs_data(obs)
            obj_data.append(obj_data_)
        obj_data = tc.concatenate(obj_data)
        return obj_data

    @torch.no_grad()
    def batched_model_predictions(self, model, images, K, obj_data, n_iterations=1, 
                                    distort=False, coarse_preds=None, predicted_gt_coarse_objects=None,
                                    multi_initializer=None):
        timer = Timer()
        timer.start()

        ids = torch.arange(len(obj_data))

        ds = TensorDataset(ids)
        dl = DataLoader(ds, batch_size=self.bsz_objects)

        preds = defaultdict(list)
        for (batch_ids, ) in dl:
            timer.resume()
            obj_inputs = obj_data[batch_ids.numpy()]
            labels = obj_inputs.infos['label'].values
            im_ids = obj_inputs.infos['batch_im_id'].values
            images_ = images[im_ids]
            K_ = K[im_ids]
            TCO_input = obj_inputs.poses
            
            outputs = model(images=images_, K=K_, TCO=TCO_input,
                            n_iterations=n_iterations, labels=labels)
            timer.pause()
            for n in range(1, n_iterations+1):
                iter_outputs = outputs[f'iteration={n}']

                infos = obj_inputs.infos
                batch_preds = tc.PandasTensorCollection(infos,
                                                        poses=iter_outputs['TCO_output'],
                                                        poses_input=iter_outputs['TCO_input'],
                                                        K_crop=iter_outputs['K_crop'],
                                                        boxes_rend=iter_outputs['boxes_rend'],
                                                        boxes_crop=iter_outputs['boxes_crop'])

                if distort:
                    assert multi_initializer is not None
                    batch_preds = multi_initializer.generate_coarse_from_angles(batch_preds)

                if self.use_gt:
                    if distort:
                        batch_preds.register_tensor(name='distortions', tensor=torch.zeros(len(batch_preds), 4).cuda()) # Add tensor to store [x, y, z, theta] values of distortions
                        # Now rewrite predictions to ground truth
                        my_counter = 0
                        while my_counter < len(batch_preds):
                            i = 0
                            found = False
                            while i < len(self.ground_truth_data) and not found:
                                pred_scene_id = batch_preds[my_counter].infos[0]
                                pred_view_id = batch_preds[my_counter].infos[1]
                                pred_obj_label = batch_preds[my_counter].infos[3]
                                for j in np.where((self.ground_truth_data[i].infos.scene_id == pred_scene_id) & (self.ground_truth_data[i].infos.view_id == pred_view_id) & (self.ground_truth_data[i].infos.label == pred_obj_label))[0]:
                                    gt_view_id = self.ground_truth_data[i][j].infos[4]
                                    gt_obj_label = self.ground_truth_data[i][j].infos[1]
                                    if gt_view_id == pred_view_id and gt_obj_label == pred_obj_label:
                                        batch_preds[my_counter].poses[0:4, 0:4] = self.ground_truth_data[i][j].poses.cuda()
                                        found = True
                                    if found:
                                        break
                                i += 1
                            # Rotate by some angle around some axis
                            x_quat, y_quat, z_quat, theta = random() - 0.5, random() - 0.5, random() - 0.5, random() * 2 * pi
                            transform = torch.tensor(pyquaternion.Quaternion(axis=[x_quat, y_quat, z_quat], angle=theta).transformation_matrix, dtype=torch.float32).cuda()
                            batch_preds[my_counter].poses[0:4, 0:4] = torch.mm(batch_preds[my_counter].poses, transform).cuda()
                            batch_preds[my_counter].distortions[0:4] = torch.tensor([x_quat, y_quat, z_quat, theta]).cuda()

                            # Append modified object to the list of GT-coarse perturbed objects
                            perturbed_object = GroundTruthPerturbationEvaluationObject(
                                distortion=[x_quat, y_quat, z_quat, theta],
                                scene_id=batch_preds[my_counter].infos[0],
                                view_id=batch_preds[my_counter].infos[1],
                                object_id=batch_preds[my_counter].infos[3],
                                detection_id=batch_preds[my_counter].infos[4]
                            )
                            predicted_gt_coarse_objects.add_object(perturbed_object)

                            my_counter += 1
                    else:
                        batch_preds.register_tensor(name="distortions", tensor=coarse_preds.distortions)
                        batch_preds.register_tensor(name="coarse_predictions", tensor=coarse_preds.poses)
                preds[f'iteration={n}'].append(batch_preds)

        logger.debug(f'Pose prediction on {len(obj_data)} detections (n_iterations={n_iterations}): {timer.stop()}')
        preds = dict(preds)
        for k, v in preds.items():
            preds[k] = tc.concatenate(v)
        return preds

    def make_TCO_init(self, detections, K):
        K = K[detections.infos['batch_im_id'].values]
        boxes = detections.bboxes
        if self.coarse_model.cfg.init_method == 'z-up+auto-depth':
            meshes = self.coarse_model.mesh_db.select(detections.infos['label'])
            points_3d = meshes.sample_points(2000, deterministic=True)
            TCO_init = TCO_init_from_boxes_zup_autodepth(boxes, points_3d, K)
        else:
            TCO_init = TCO_init_from_boxes(z_range=(1.0, 1.0), boxes=boxes, K=K)
        return tc.PandasTensorCollection(infos=detections.infos, poses=TCO_init)

    def get_predictions(self, images, K,
                        detections=None,
                        data_TCO_init=None,
                        n_coarse_iterations=1,
                        n_refiner_iterations=1,
                        predicted_gt_coarse_objects=None,
                        multi_initializer=None):

        print("Start predicting...")
        preds = dict()
        if data_TCO_init is None:
            assert detections is not None
            assert self.coarse_model is not None
            assert n_coarse_iterations > 0
            data_TCO_init = self.make_TCO_init(detections, K)
            coarse_preds = self.batched_model_predictions(self.coarse_model,
                                                          images, K, data_TCO_init,
                                                          n_iterations=n_coarse_iterations,
                                                          distort=False,
                                                          predicted_gt_coarse_objects=predicted_gt_coarse_objects,
                                                          multi_initializer=multi_initializer)
            for n in range(1, n_coarse_iterations + 1):
                preds[f'coarse/iteration={n}'] = coarse_preds[f'iteration={n}']
            data_TCO = coarse_preds[f'iteration={n_coarse_iterations}']
        else:
            assert n_coarse_iterations == 0
            data_TCO = data_TCO_init
            preds[f'external_coarse'] = data_TCO

        if n_refiner_iterations >= 1:
            assert self.refiner_model is not None
            refiner_preds = self.batched_model_predictions(self.refiner_model,
                                                           images, K, data_TCO,
                                                           n_iterations=n_refiner_iterations,
                                                           distort=False,
                                                           coarse_preds=coarse_preds['iteration=1'])
            for n in range(1, n_refiner_iterations + 1):
                preds[f'refiner/iteration={n}'] = refiner_preds[f'iteration={n}']
            data_TCO = refiner_preds[f'iteration={n_refiner_iterations}']
        return data_TCO, preds
