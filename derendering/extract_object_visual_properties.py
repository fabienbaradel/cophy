"""
Given a pre-trained model it extract the object properties (presence, bounding boxes and pose 3D)

# debugging:
ipython derendering/extract_object_visual_properties.py --sanity_check

"""

from dataloaders.dataset_collision import Collision_CF
from dataloaders.dataset_blocktower import Blocktower_CF
from dataloaders.dataset_balls import Balls_CF
from derendering.model import DeRendering
from torch import optim
from torch.utils.data import DataLoader
import torch
import numpy as np
import argparse
import ipdb
import os
from tqdm import *
from random import choice
import torch.nn.functional as F
import time
from dataloaders.utils import *
import pickle as pkl


def extract_object_visual_properties(model, device, loader, sanity_check=False):
  model.eval()

  dict_id2prop = {}
  for i, input in enumerate(tqdm(loader)):
    # select at timestep and rgb, presence, pose_2D and pose_3D
    B, T, C, H, W = input['rgb_cd'].shape
    assert B == 1
    rgb_cd = input['rgb_cd'].view(T, C, H, W).to(device)
    rgb_ab = input['rgb_ab'].view(T, C, H, W).to(device)
    rgb = torch.cat([rgb_ab, rgb_cd], 0)

    # forward
    pred_presence, pred_pose_3D, pred_pose_2D = model(rgb)
    pred_presence = (pred_presence > 0).float()
    pred_pose_3D = pred_pose_3D * pred_presence.unsqueeze(-1)
    pred_pose_2D = pred_pose_2D * pred_presence.unsqueeze(-1)
    pred_presence = pred_presence.detach().cpu().numpy()
    pred_pose_3D = pred_pose_3D.detach().cpu().numpy()
    pred_pose_2D = pred_pose_2D.detach().cpu().numpy()

    # flatten
    pred_presence_ab, pred_presence_cd = pred_presence[0], pred_presence[T]
    pred_pose_3d_ab, pred_pose_3d_cd = pred_pose_3D[:T].flatten(
      'C'), pred_pose_3D[T:].flatten('C')
    pred_pose_2d_ab, pred_pose_2d_cd = pred_pose_2D[:T].flatten(
      'C'), pred_pose_2D[T:].flatten('C')

    prop = np.concatenate([pred_presence_ab, pred_pose_2d_ab, pred_pose_3d_ab,
                           pred_presence_cd, pred_pose_2d_cd, pred_pose_3d_cd,
                           ], 0)

    dict_id2prop[input['id'][0]] = prop.astype(np.float16)

    if sanity_check:
      # make sure that we can re-arrange the flatten array
      prop_ab, prop_cd = np.split(prop, 2)
      K = loader.dataset.num_objects
      presence = prop_ab[:K]
      pose_2d = prop_ab[K:(K + T * K * 4)].reshape(T, K, 4)
      pose_3d = prop_ab[(K + T * K * 4):].reshape(T, K, 3)
      add_bboxes_to_video(rgb_ab.detach().cpu().numpy(),
                          pose_2d,
                          presence,
                          loader.dataset.colors,
                          out_fn=os.path.join(f"./rgb_pred_bboxes_{i:04d}.mp4"))
      os._exit(0)


  return dict_id2prop


def get_dataloaders(dataset_name, dataset_dir, kwargs_loader, split, type,
    num_objects):
  # choice of dataset
  if dataset_name == 'balls':
    dataset = Balls_CF(num_balls=num_objects,
                       root_dir=dataset_dir,
                       split=split,
                       is_rgb=True, only_cd=False)
    fn = os.path.join(f"balls_{args.num_objects}_{args.split}")
  elif dataset_name == 'collision':
    dataset = Collision_CF(type=type,
                           root_dir=dataset_dir,
                           split=split,
                           is_rgb=True, only_cd=False)
    fn = os.path.join(f"collision_{args.type}_{args.split}")
  elif dataset_name == 'blocktower':
    dataset = Blocktower_CF(type=type,
                            num_blocks=num_objects,
                            root_dir=dataset_dir,
                            split=split,
                            is_rgb=True, only_cd=False)
    fn = os.path.join(f"blocktower_{args.num_objects}_{args.type}_{args.split}")
  else:
    raise NameError('Unkown dataset name.')

  # loader
  kwargs_loader['batch_size'] = 1
  loader = DataLoader(dataset, **kwargs_loader)

  return loader, fn


def main(args):
  # kwargs
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  kwargs_loader = {}
  if device.type == 'cuda':
    kwargs_loader.update({'num_workers': args.workers, 'pin_memory': True})

  # datasets and loaders
  loader, fn = get_dataloaders(args.dataset_name,
                               args.dataset_dir,
                               kwargs_loader,
                               args.split,
                               args.type,
                               args.num_objects)

  # model and optim
  model = DeRendering(num_objects=loader.dataset.num_objects).to(device)

  # load the derendering module
  pretrained_dict = torch.load(args.derendering_ckpt)
  pretrained_dict = {k: v for k, v in pretrained_dict.items()}
  model_dict = model.state_dict()
  pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                     k in model_dict}
  model.load_state_dict(pretrained_dict, strict=False)

  # extract
  dict_id2object_properties = extract_object_visual_properties(model, device, loader, args.sanity_check)

  # save
  os.makedirs(args.out_dir, exist_ok=True)
  out_fn = os.path.join(args.out_dir, f"{fn}_extracted_prop.pickle")
  with open(out_fn, 'wb') as f:
    pkl.dump(dict_id2object_properties, f, protocol=pkl.HIGHEST_PROTOCOL)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description='Training of the derendering module.')
  parser.add_argument('--dataset_dir',
                      default='/tmp/CoPhy_224/blocktowerCF',
                      type=str,
                      help='Location of the data.')
  parser.add_argument('--derendering_ckpt',
                      # default='/usr/local/google/home/fbaradel/Documents/log_dir/ballsCF/model_state_dict.pt',
                      default='./ckpts/derendering/blocktowerCF/model_state_dict.pt',
                      type=str,
                      help='Location of the pre-trained derendering module.')
  parser.add_argument('--out_dir',
                      default='./preextracted_object_visual_properties',
                      type=str,
                      help='Location of the out dir.')
  parser.add_argument('--dataset_name',
                      default='blocktower',
                      type=str,
                      help='Which dataset to take (balls, collision, blocktower).')
  parser.add_argument('--workers', default=8, type=int, help='Workers.')
  parser.add_argument('--num_objects',
                      default=4,
                      type=int,
                      help='Number of objects for training.')
  parser.add_argument('--type',
                      default='normal',
                      type=str,
                      help='Type of train/val/test split.')
  parser.add_argument('--split',
                      default='val',
                      type=str,
                      help='Type of train/val/test split.')
  parser.add_argument('--sanity_check', dest='sanity_check', action='store_true')
  parser.add_argument('--no_sanity_check', dest='sanity_check', action='store_false')
  parser.set_defaults(feature=False)
  args = parser.parse_args()

  main(args)
