import pdb
import cosypose.utils.tensor_collection as tc
import pandas as pd
import torch

class GroundTruthPerturbationEvaluationObject():
    """
    This class records errors associated with perturbed ground-truth predictions.
    Errors are recorded in the following format:
    errors = {
        'object_1': [[[x,y,z,theta], error], [[x,y,z,theta], error], ...], 
        'object_2': [[[x,y,z,theta], error], [[x,y,z,theta], error], ...],
        ...
        }
    """
    def __init__(self,
                distortion,
                scene_id,
                view_id,
                object_id,
                detection_id,
                coarse_error=None, 
                refiner_error=None,
                ):
        self.distortion = distortion
        self.scene_id = scene_id
        self.view_id = view_id
        self.object_id = object_id
        self.detection_id = detection_id
        self.coarse_error = coarse_error
        self.refiner_error = refiner_error

    def __str__(self) -> str:
        text = f"Distortion = {self.distortion}\nScene_id, view_id, detection_id, object_id = {self.scene_id}, {self.view_id}, {self.detection_id}, {self.object_id}\nCoarse error value: {self.coarse_error}\nRefiner error value: {self.refiner_error}"
        return text

    def update_coarse_error(self, coarse_error):
        self.coarse_error = coarse_error
    
    def update_refiner_error(self, refiner_error):
        self.refiner_error = refiner_error
    

class GroundTruthPerturbationEvaluationArray():
    """
    This class stores an array of objects of class GroundTruthPerturbationEvaluationObject.
    """
    def __init__(self, objects=list()) -> None:
        self.objects = objects

    def __str__(self) -> str:
        text = f"Total number of objects recorded: {self.length()}."
        return text
    
    def __getitem__(self, i):
        return self.objects[i]

    def add_object(self, new_object):
        self.objects.append(new_object)
    
    def length(self):
        return len(self.objects)
    
    def get_objects(self):
        return self.objects

    def multi_initialization(self, batch_preds, n_repeats):
        """
        Loop through each objects.
            Loop n times.
                Record each i-th measurement, updating the detection id.
        """
        new_infos = pd.DataFrame(columns=batch_preds.infos.columns)
        new_infos = new_infos.astype({
            'scene_id': 'int',
            'view_id': 'int',
            'score': 'float',
            'label': 'object',
            'det_id': 'int',
            'batch_im_id': 'int',
            'group_id': 'int'
        })

        new_poses = torch.empty(size=(0, 4, 4), dtype=torch.float32).cuda()
        new_poses_input = torch.empty(size=(0, 4, 4), dtype=torch.float32).cuda()
        new_K_crop = torch.empty(size=(0, 3, 3), dtype=torch.float32).cuda()
        new_boxes_rend = torch.empty(size=(0, 4), dtype=torch.float32).cuda()
        new_boxes_crop = torch.empty(size=(0, 4), dtype=torch.float32).cuda()

        idx = 0
        for obj in batch_preds:
            for i in range(n_repeats):
                # Update new_infos
                infos_data = obj.infos.tolist()
                new_obj = pd.DataFrame([infos_data], columns=new_infos.columns, index=[str(idx)])
                new_infos = new_infos.append(new_obj, ignore_index=True)

                # Update torch tensors
                new_poses = torch.cat((new_poses, torch.unsqueeze(obj.poses, dim=0)), 0)
                new_poses_input = torch.cat((new_poses_input, torch.unsqueeze(obj.poses_input, dim=0)), 0)
                new_K_crop = torch.cat((new_K_crop, torch.unsqueeze(obj.K_crop, dim=0)), 0)
                new_boxes_rend = torch.cat((new_boxes_rend, torch.unsqueeze(obj.boxes_rend, dim=0)), 0)
                new_boxes_crop = torch.cat((new_boxes_crop, torch.unsqueeze(obj.boxes_crop, dim=0)), 0)

                idx += 1

        new_batch_preds = tc.PandasTensorCollection(new_infos,
                                                    poses=new_poses,
                                                    poses_input=new_poses_input,
                                                    K_crop=new_K_crop,
                                                    boxes_rend=new_boxes_rend,
                                                    boxes_crop=new_boxes_crop)
        return new_batch_preds