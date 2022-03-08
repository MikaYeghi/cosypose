import argparse
import numpy as np
import os
from colorama import Fore, Style

# from cosypose.training.train_pose import train_pose
from cosypose.registering.register_object import train_pose
from cosypose.utils.logging import get_logger
logger = get_logger(__name__)


def make_cfg(args):
    cfg = argparse.ArgumentParser('').parse_args([])
    if args.config:
        logger.info(f"{Fore.GREEN}Training with config: {args.config} {Style.RESET_ALL}")

    cfg.resume_run_id = None
    if len(args.resume) > 0:
        cfg.resume_run_id = args.resume
        logger.info(f"{Fore.RED}Resuming {cfg.resume_run_id} {Style.RESET_ALL}")

    N_CPUS = int(os.environ.get('N_CPUS', 10))
    N_WORKERS = min(N_CPUS - 2, 8)
    N_WORKERS = 8
    N_RAND = np.random.randint(1e6)

    run_comment = ''

    # Data
    cfg.urdf_ds_name = 'ycbv'
    cfg.object_ds_name = 'ycbv.bop-compat'
    cfg.n_symmetries_batch = 64

    cfg.train_ds_names = [('synt.ycbv-1M', 1), ('ycbv.real.train', 3), ('ycbv.synthetic.train', 3)]
    cfg.val_ds_names = cfg.train_ds_names
    cfg.val_epoch_interval = 10
    cfg.test_ds_names = ['ycbv.test.keyframes', ]
    cfg.test_epoch_interval = 30
    cfg.n_test_frames = None

    cfg.input_resize = (480, 640)
    cfg.rgb_augmentation = True
    cfg.background_augmentation = True
    cfg.gray_augmentation = False

    # Model
    cfg.backbone_str = 'efficientnet-b3'
    cfg.run_id_pretrain = None
    cfg.n_pose_dims = 9
    cfg.n_rendering_workers = N_WORKERS
    cfg.refiner_run_id_for_test = None
    cfg.coarse_run_id_for_test = None

    # Optimizer
    cfg.lr = 3e-4
    cfg.weight_decay = 0.
    cfg.n_epochs_warmup = 50
    cfg.lr_epoch_decay = 500
    cfg.clip_grad_norm = 0.5

    # Training
    cfg.batch_size = 32
    cfg.epoch_size = 115200
    cfg.n_epochs = 700
    cfg.n_dataloader_workers = N_WORKERS

    # Method
    cfg.loss_disentangled = True
    cfg.n_points_loss = 2600
    cfg.TCO_input_generator = 'fixed'
    cfg.n_iterations = 1
    cfg.min_area = None

    if 'tless-' in args.config:
        cfg.urdf_ds_name = 'tless.cad'
        cfg.object_ds_name = 'tless.cad'
        cfg.train_ds_names = [('synthetic.tless-1M.train', 1),
                              ('tless.primesense.train', 5)]
        cfg.val_ds_names = [('synthetic.tless-1M.val', 1)]

        cfg.test_ds_names = ['tless.primesense.test', ]
        cfg.input_resize = (540, 720)

        if args.config == 'tless-unseen-registration':
            cfg.train_ds_names = [('tless.register.object', 5)]
            cfg.val_ds_names = [('tless.seen.dataset', 1)]
            cfg.test_ds_names = []
            cfg.n_epochs = 170
            cfg.lr_epoch_decay = 500
            cfg.n_epochs_warmup = 50
            # cfg.val_epoch_interval = 1
            cfg.batch_size = 1
            cfg.epoch_size = 11520
            cfg.background_augmentation = False 
            cfg.rgb_augmentation = False
            cfg.n_dataloader_workers = 8 
            cfg.n_rendering_workers = 8
            # cfg.TCO_input_generator = 'gt+noise'
            cfg.TCO_input_generator = 'fixed'
            cfg.renderer = 'pytorch3d'
            cfg.features_on = True
            cfg.n_feature_channels = 64
            cfg.features_dict = 'object-features-85446619504502821800'
            cfg.resume_run_id = 'tless-coarse-new--107780'
            cfg.lr = 1.0e-4

        else:
            raise ValueError(args.config)

    else:
        raise ValueError(args.config)

    if args.no_eval:
        cfg.test_ds_names = []

    cfg.run_id = f'{args.config}-{run_comment}-{N_RAND}'

    if args.debug:
        cfg.test_ds_names = []
        cfg.n_epochs = 4
        cfg.val_epoch_interval = 1
        cfg.batch_size = 4
        cfg.epoch_size = 4 * cfg.batch_size
        cfg.run_id = 'debug-' + cfg.run_id
        cfg.background_augmentation = True
        cfg.n_dataloader_workers = 8
        cfg.n_rendering_workers = 0
        cfg.n_test_frames = 10

    N_GPUS = int(os.environ.get('N_PROCS', 1))
    cfg.epoch_size = cfg.epoch_size // N_GPUS
    return cfg


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training')
    parser.add_argument('--config', default='', type=str)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--no-eval', action='store_true')
    parser.add_argument('--resume', default='', type=str)
    args = parser.parse_args()

    cfg = make_cfg(args)
    train_pose(cfg)
