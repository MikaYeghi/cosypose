import torch
import os
from pytorch3d.io import load_ply
from cosypose.config import BOP_DS_DIR, FEATURES_DIR
import pickle
import random
import string
import pdb

CAD_DS_DIR = BOP_DS_DIR / 'tless' / 'models_cad' # primary path to the dataset of CAD models

def get_random_code(length):
    digits = string.digits
    result_code = ''.join(random.choice(digits) for i in range(length))
    return result_code

class FeatureLoader:
    def __init__(self, ds_path=CAD_DS_DIR, features_dict=None, code_length=20, number_of_channels=32) -> None:
        self.ds_path = ds_path
        self.code_length = code_length

        # Extract object feature tensors
        self.features = self.extract_features(ds_path, features_dict=features_dict, number_of_channels=number_of_channels)
        self.number_of_channels = number_of_channels

    def extract_features(self, ds_path=CAD_DS_DIR, number_of_channels=32, cad_suffix='.ply', features_dict=None):
        if features_dict is None:
            obj_path = list()
            features_ = dict()

            for filename in os.listdir(ds_path):
                if filename.endswith(cad_suffix):
                    full_path = os.path.join(ds_path, filename)
                    obj_path.append(full_path)
            obj_path.sort()

            for obj in obj_path:
                obj_label = obj.split('/')[-1].split('.')[0]
                verts, _ = load_ply(obj) # loads a tuple of (vertices, features)
                features__ = torch.randn(size=(verts.shape[0], number_of_channels)) # torch.nn.Parameter(...); divide by the sqrt of num of channels
                features_[obj_label] = features__
        else:
            features_path = FEATURES_DIR / (features_dict + '.pkl') # features dictionary must be a pkl file
            if features_path.exists():
                with open(features_path, 'rb') as f:
                    try:
                        features_ = pickle.load(f)
                    except Exception:
                        raise ValueError(f"Incorrect data format in file '{features_path}'")
            else:
                raise ValueError(f"Features dictionary '{features_path}' not found")
        return features_ # torch.nn.ModuleDict(...)

    def get_features(self):
        return self.features

    def get_n_channels(self):
        return self.number_of_channels

    def save_features(self):
        code = get_random_code(self.code_length)
        save_dir = FEATURES_DIR / f'object-features-{code}.pkl'
        with open(str(save_dir), 'wb') as f:
            pickle.dump(self.get_features(), f, protocol=pickle.HIGHEST_PROTOCOL)