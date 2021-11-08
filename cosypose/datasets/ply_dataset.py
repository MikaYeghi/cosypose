import pandas as pd
from pathlib import Path

import pdb


class PlyDataset:
    def __init__(self, ds_dir, sort_indices=False):
        ds_dir = Path(ds_dir)
        index = []
        for ply_path in Path(ds_dir).iterdir():
            if ply_path.suffix == '.ply':
                infos = dict(
                    label=ply_path.stem,
                    ply_path=ply_path.as_posix(),
                    scale=1.0,
                )
                index.append(infos)
        self.index = pd.DataFrame(index)
        if sort_indices:
            self.sort_index() 

    def __getitem__(self, idx):
        return self.index.iloc[idx]

    def __len__(self):
        return len(self.index)

    def sort_index(self):
        self.index = self.index.sort_values('label')
        self.index = self.index.reset_index(drop=True)

    def get_ply_path_by_label(self, label):
        obj_data = self.index.loc[self.index['label'] == label]
        obj_path = obj_data.iloc[0]['ply_path']
        scale = obj_data.iloc[0]['scale']
        return obj_path, scale



class BOPPlyDataset(PlyDataset):
    def __init__(self, ds_dir):
        super().__init__(ds_dir)
        self.index['scale'] = 0.001