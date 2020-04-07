"""
BallsCF dataloader.

# Command line for a sanity check:
ipython dataloaders/dataset_balls.py
"""


import ipdb
from tqdm import tqdm
import os
import numpy as np
import argparse
import random
import torch.utils.data as data
import imageio
from dataloaders.utils import *
import pickle

COLORS = ['red', 'green', 'blue', 'yellow', 'lime', 'purple', 'orange', 'cyan', 'magenta']
T=30
MASSES = [1, 2, 5]
FRICTIONS = [0.1, 0.5, 1]
RESTITUTIONS = [0.1, 0.5, 1]

class Balls_CF(data.Dataset):
  """
    Dataset size (train/val/test):
    - 2: 7k/2k/1k
    - 3: 7k/2k/1k
    - 4: 7k/2k/1k
    - 5: 7k/2k/1k
    - 6: 7k/2k/1k
  """
  def __init__(self,
      num_balls,
      split,
      root_dir,
      is_rgb=False,
      only_cd=True,
      pose_3d_fn='states.npy',
      use_preextracted_object_properties=False,
      preextracted_object_properties_dir='',
      num_examples=None,
      *args, **kwargs,
  ):
    self.pose_3d_fn = pose_3d_fn
    self.only_cd = only_cd
    self.num_balls = num_balls
    self.split = split
    self.is_rgb = is_rgb
    self.is_training = True if split == 'train' else False
    self.root_dir = root_dir
    self.num_objects = len(COLORS)
    self.colors = COLORS
    self.use_preextracted_object_properties = use_preextracted_object_properties
    self.preextracted_object_properties_dir = preextracted_object_properties_dir

    with open(f"./dataloaders/splits/ballsCF_{self.split}_{self.num_balls}.txt", 'r') as f:
      lines = f.readlines()
    self.list_ex = [x.strip() for x in lines]
    if num_examples:
        self.list_ex = self.list_ex[:num_examples]

    if self.use_preextracted_object_properties:
      fn = os.path.join(self.preextracted_object_properties_dir, f"balls_{self.num_balls}_{self.split}_extracted_prop.pickle")
      with open(fn, 'rb') as f:
        self.dict_id2object_properties = pickle.load(f)

    print(self.__repr__())

  def __len__(self):
    return len(self.list_ex)

  def __repr__(self):
    return f"BallsCF: Number of balls={self.num_balls} - Split={self.split} - N={self.__len__()}"

  def __getitem__(self, item):
    """ Extract one example of the dataset """

    # Sequences
    ex = self.list_ex[item]
    cd = os.path.join(self.root_dir, str(self.num_balls), ex, 'cd')
    ab = os.path.join(self.root_dir, str(self.num_balls), ex, 'ab')

    # 3D - do not take into account the z coordinates
    pose_3D_cd, presence_cd = get_pose_3D(cd, fn=self.pose_3d_fn) # (T, K, 3)
    pose_3D_ab, presence_ab = get_pose_3D(ab, fn=self.pose_3d_fn)
    stab_cd = get_stab(pose_3D_cd, presence_cd, t_delta=2, eps=0.05) # (T,K)
    stab_ab = get_stab(pose_3D_ab, presence_ab, t_delta=2, eps=0.05)
    num_objects_ab = int(presence_ab.sum())
    num_objects_cd = int(presence_cd.sum())

    # 2D
    pose_2D_cd = get_pose_2D(cd) # (T, K, 4)
    pose_2D_ab = get_pose_2D(ab)

    # Confounders
    confounders = get_confounders(cd) # (K,mass-friction-restitution)

    out = {
        'pose_3D_ab': pose_3D_ab,
        'pose_3D_cd': pose_3D_cd,
        'pose_2D_ab': pose_2D_ab,
        'pose_2D_cd': pose_2D_cd,
        'presence_ab': presence_ab,
        'presence_cd': presence_cd,
        'stab_ab': stab_ab,
        'stab_cd': stab_cd,
        'confounders': confounders,
        'num_objects_ab': num_objects_ab,
        'num_objects_cd': num_objects_cd,
        'id': ex
    }
    # Pixels
    if self.is_rgb:
      out['rgb_cd'] = get_rgb(cd)

      if not self.only_cd:
        out['rgb_ab'] = get_rgb(ab)

    # Pre-computed object properties
    if self.use_preextracted_object_properties:
      prop = self.dict_id2object_properties[ex]
      prop_ab, prop_cd = np.split(prop, 2)
      K = self.num_objects

      # AB
      out['pred_presence_ab'] = prop_ab[:K].astype(np.float32)
      out['pred_pose_2D_ab'] = prop_ab[K:(K + T * K * 4)].reshape(T, K, 4).astype(np.float32)
      out['pred_pose_3D_ab'] = prop_ab[(K + T * K * 4):].reshape(T, K, 3).astype(np.float32)

      # CD
      out['pred_presence_cd'] = prop_cd[:K].astype(np.float32)
      out['pred_pose_2D_cd'] = prop_cd[K:(K + T * K * 4)].reshape(T, K, 4).astype(np.float32)
      out['pred_pose_3D_cd'] = prop_cd[(K + T * K * 4):].reshape(T, K, 3).astype(np.float32)

    return out


if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='Dataloader for BallsCF.')
  parser.add_argument('--root_dir',
                      default='/storage/Datasets/CoPhy/CoPhy_224/ballsCF',
                      type=str,
                      help='Location of the data.')
  args = parser.parse_args()

  dataset = Balls_CF(num_balls=2,
                     root_dir=args.root_dir,
                     split='train',
                     is_rgb=True,
                     only_cd=False,
                     )

  # Drawings the bounding boxes on the video
  ex = dataset.__getitem__(0)
  add_bboxes_to_video(ex['rgb_ab'], ex['pose_2D_ab'], ex['presence_ab'], COLORS, out_fn='./rgb_balls_ab_w_bboxes.mp4')
  add_bboxes_to_video(ex['rgb_cd'], ex['pose_2D_cd'], ex['presence_cd'], COLORS, out_fn='./rgb_balls_cd_w_bboxes.mp4')
  print(f"Have a look at the nex mp4 files created on your workspace.")
