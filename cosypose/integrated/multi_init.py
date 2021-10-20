import pandas as pd
import torch
import cosypose.utils.tensor_collection as tc
from tqdm import tqdm
import pdb
import scipy
from itertools import product
import numpy as np
from scipy.spatial.transform import Rotation
from pyquaternion import Quaternion

class MultipleInitializer():
    def __init__(self, objects=list(),
                    multi_initializations=None,
                    n_repeats=None) -> None:
        self.objects = objects
        self.multi_initializations = multi_initializations
        self.n_repeats = n_repeats

    def generate_multi_initialization(self, detections, n_repeats):
        """
        Loop through each object.
            Loop n times.
                Record each i-th measurement, updating the detection id.
        """

        # Process base data
        columns = list(detections.infos.columns)
        columns.append('x_angle')
        columns.append('y_angle')
        columns.append('z_angle')
        new_infos = pd.DataFrame(columns=columns)
        new_infos = new_infos.astype({
            'scene_id': 'int',
            'view_id': 'int',
            'score': 'float',
            'label': 'object'
        })

        # Angles generation
        n_initial_objects = len(detections)
        n_angles = round(n_repeats ** (1 / 3))
        base_angles = [i * 360 / (n_angles) for i in range(n_angles)]
        angles = list(product(base_angles, base_angles, base_angles))
        self.n_repeats = n_repeats

        # Initialize large arrays
        assert len(detections) > 0
        infos_data = [None for _ in range(len(detections[0].infos.tolist()) + 3)]
        new_infos_array = [[] for _ in range(n_initial_objects * n_repeats)]
        new_poses = torch.empty(size=(n_initial_objects * n_repeats, 4, 4), dtype=torch.float64).cpu()
        new_bboxes = torch.empty(size=(n_initial_objects * n_repeats, 4), dtype=torch.float32).cpu()

        print(f"Multi-initializer: starting {n_initial_objects * n_repeats} initializations...")
        idx = 0
        for obj in tqdm(detections):
            for i in range(n_repeats):
                # Update new_infos
                infos_data[:-3] = obj.infos.tolist()
                x_angle = angles[i][0]
                y_angle = angles[i][1]
                z_angle = angles[i][2]
                infos_data[-3] = x_angle
                infos_data[-2] = y_angle
                infos_data[-1] = z_angle
                new_infos_array[idx] = infos_data.copy()

                # Update torch tensors
                new_poses[idx] = obj.poses
                new_bboxes[idx] = obj.bboxes

                idx += 1

        new_infos = pd.DataFrame(new_infos_array, columns=columns)
        multi_detections = tc.PandasTensorCollection(new_infos,
                                                    poses=new_poses,
                                                    bboxes=new_bboxes)
        self.multi_initializations = multi_detections
        print(f"Multi-initializer: initializations generated. Total number of objects: {len(multi_detections)}")
        return multi_detections
    
    def generate_sliced_multi_initialization(self, detections, n_repeats_per_axis):
        # Process base data
        columns = list(detections.infos.columns)
        columns.append('x_angle')
        columns.append('y_angle')
        columns.append('z_angle')
        new_infos = pd.DataFrame(columns=columns)
        new_infos = new_infos.astype({
            'scene_id': 'int',
            'view_id': 'int',
            'score': 'float',
            'label': 'object'
        })

        # Generate angles for a slice
        n_initial_objects = len(detections)
        angles_per_axis = np.linspace(-np.pi, np.pi, n_repeats_per_axis)
        angle_combinations = list(product(angles_per_axis, [0], angles_per_axis))
        angle_combinations_filtered = [angle for angle in angle_combinations if np.sum(np.array(angle) ** 2) < np.pi ** 2]
        n_initializations = len(angle_combinations_filtered)

        # Initialize large arrays
        assert len(detections) > 0
        infos_data = [None for _ in range(len(detections[0].infos.tolist()) + 3)]
        new_infos_array = [[] for _ in range(n_initial_objects * n_initializations)]
        new_poses = torch.empty(size=(n_initial_objects * n_initializations, 4, 4), dtype=torch.float64).cpu()
        new_bboxes = torch.empty(size=(n_initial_objects * n_initializations, 4), dtype=torch.float32).cpu()
        # pdb.set_trace()
        print(f"Multi-initializer: starting {n_initial_objects * n_initializations} initializations...")
        idx = 0
        for obj in tqdm(detections):
            for i in range(n_initializations):
                # Update new_infos
                infos_data[:-3] = obj.infos.tolist()
                x_angle = angle_combinations_filtered[i][0]
                y_angle = angle_combinations_filtered[i][1]
                z_angle = angle_combinations_filtered[i][2]
                infos_data[-3] = x_angle
                infos_data[-2] = y_angle
                infos_data[-1] = z_angle
                new_infos_array[idx] = infos_data.copy()

                # Update torch tensors
                new_poses[idx] = obj.poses
                new_bboxes[idx] = obj.bboxes

                idx += 1

        new_infos = pd.DataFrame(new_infos_array, columns=columns)
        multi_detections = tc.PandasTensorCollection(new_infos,
                                                    poses=new_poses,
                                                    bboxes=new_bboxes)
        self.multi_initializations = multi_detections
        print(f"Multi-initializer: initializations generated. Total number of objects: {len(multi_detections)}")
        return multi_detections

    def get_multi_initializations(self):
        return self.multi_initializations
    
    def get_n_repeats(self):
        return self.n_repeats
    
    def generate_coarse_from_angles(self, preds):
        for obj_idx in range(len(preds)):
            x_angle = preds.infos['x_angle'][obj_idx]
            y_angle = preds.infos['y_angle'][obj_idx]
            z_angle = preds.infos['z_angle'][obj_idx]
            # rotation = Rotation.from_euler('xyz', angles=[x_angle, y_angle, z_angle], degrees=False)
            # rotation_matrix = rotation.as_matrix()
            rotation = Quaternion(axis=[x_angle, y_angle, z_angle], angle=np.sqrt(x_angle ** 2 + y_angle ** 2 + z_angle ** 2))
            rotation_matrix = rotation.transformation_matrix[0:3, 0:3]
            preds[obj_idx].poses[0:3, 0:3] = torch.tensor(rotation_matrix)
        
        return preds
    
    def pick_object(self, detections, object_label, scene_id=None, view_id=None):
        if scene_id is not None and view_id is not None:
            obj_idx = detections.infos.index[(detections.infos['scene_id'] == scene_id) & (detections.infos['view_id'] == view_id) & (detections.infos['label'] == object_label)]
        else:
            obj_idx = detections.infos.index[detections.infos['label'] == object_label]
        infos = pd.DataFrame(detections.infos.iloc[obj_idx], columns=detections.infos.columns)
        poses = detections.poses[obj_idx]
        bboxes = detections.bboxes[obj_idx]

        single_object_detection = tc.PandasTensorCollection(infos,
                                                            poses=poses,
                                                            bboxes=bboxes)
        self.multi_initializations = single_object_detection
        return self.multi_initializations