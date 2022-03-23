from cosypose.rendering.bullet_batch_renderer import BulletBatchRenderer
from cosypose.rendering.pytorch3d_scene_renderer import Pytorch3DSceneRenderer
import pdb

def make_renderer(args, device):
    if args.renderer == 'pybullet':
        renderer = BulletBatchRenderer(object_set=args.urdf_ds_name, n_workers=args.n_rendering_workers)
    elif args.renderer == 'pytorch3d':
        renderer = Pytorch3DSceneRenderer(ply_ds=args.urdf_ds_name, 
                                        device=device, 
                                        n_feature_channels=args.n_feature_channels, 
                                        features_on=args.features_on, 
                                        features_dict=args.features_dict,
                                        save_dir=args.save_dir)
    else:
        raise NotImplementedError
    return renderer    