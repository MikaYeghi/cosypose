from cosypose.utils.logs_bokeh import Plotter
from cosypose.config import LOCAL_DATA_DIR
from pathlib import Path
log_dir = Path(LOCAL_DATA_DIR / 'experiments')
import os
import pdb

run_ids = [
    'tless-coarse--10219',
    'tless-refiner--585928',
    'tless-coarse-new--85295',
    'tless-refiner-new--914693',
    'tless-coarse-new--210857',
    'tless-refiner-new--156077',
    'tless-coarse-new--210857-object25-without-MSE',
    'tless-refiner-new--156077-object25-without-MSE',
    'tless-coarse-new--210857-object25-with-MSE',
    'tless-refiner-new--156077-object25-with-MSE',
    'tless-coarse-new--210857-object25-rough',
    'tless-refiner-new--156077-object25-rough',
    'tless-coarse-new--861929',
    'tless-coarse-new--759279'
]
save_dir = '/home/yemika/Mikael/Oxford/Studying/4YP/code/graph_results'

plotter = Plotter(log_dir)
plotter.load_logs(run_ids)

for run_id in run_ids:
    data = plotter.log_dicts[run_id]['train_loss_total'].tolist()
    save_path = os.path.join(save_dir, run_id + '.txt')
    with open(save_path, 'w') as f:
        for data_point in data:
            f.write(str(data_point) + '\n')