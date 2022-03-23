import torch
import torch.nn as nn

import os
from pytorch3d.io import load_ply
from cosypose.config import BOP_DS_DIR, FEATURES_DIR
from cosypose.utils.logging import get_logger
import pickle
import random
import string
import pdb

CAD_DS_DIR = BOP_DS_DIR / 'tless' / 'models_cad' # primary path to the dataset of CAD models

logger = get_logger(__name__)

def get_random_code(length):
    digits = string.digits
    result_code = ''.join(random.choice(digits) for _ in range(length))
    return result_code

class FeatureLoader(nn.Module):
    def __init__(self, save_dir, ds_path=CAD_DS_DIR, features_dict=None, code_length=20, number_of_channels=32) -> None:
        super(FeatureLoader, self).__init__()
        self.ds_path = ds_path
        self.code_length = code_length
        self.features_dict = features_dict

        # Extract object feature tensors
        self.features = self.extract_features(save_dir=save_dir, ds_path=ds_path, features_dict=features_dict, number_of_channels=number_of_channels)
        self.number_of_channels = number_of_channels

    def extract_features(self, save_dir, ds_path=CAD_DS_DIR, number_of_channels=32, cad_suffix='.ply', features_dict=None):
        if features_dict is None:
            logger.info("Initializing random features for objects...")
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
                features__ = nn.parameter.Parameter(torch.randn(size=(verts.shape[0], number_of_channels)), requires_grad=True) # [MIKAEL] divide by the sqrt of num of channels
                features_[obj_label] = features__
        else:
            features_path = save_dir / (features_dict + '.pkl') # features dictionary must be a pkl file
            logger.info(f"Loading object features from {features_path}")
            if features_path.exists():
                with open(features_path, 'rb') as f:
                    try:
                        features_ = pickle.load(f)
                    except Exception:
                        raise ValueError(f"Incorrect data format in file '{features_path}'")
            else:
                raise ValueError(f"Features dictionary '{features_path}' not found")
        features_ = nn.modules.ParameterDict(features_)
        # save_dir = FEATURES_DIR / f'object-features-initial.pkl'
        # with open(str(save_dir), 'wb') as f:
        #     pickle.dump(features_, f, protocol=pickle.HIGHEST_PROTOCOL)
        return features_

    def get_features(self):
        return self.features

    def get_n_channels(self):
        return self.number_of_channels

    def save_features(self, save_dir, verbose=1):
        if self.features_dict is not None:
            code = self.features_dict
            save_dir = save_dir / f'{code}.pkl'
        else:
            code = get_random_code(self.code_length)
            self.features_dict = code
            save_dir = save_dir / f'object-features-{code}.pkl'
        with open(str(save_dir), 'wb') as f:
            pickle.dump(self.get_features(), f, protocol=pickle.HIGHEST_PROTOCOL)
        if verbose != 0:
            print(f"Saving features to {save_dir}.")