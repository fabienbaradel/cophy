"""
CollisionCF data loader

ipython dataloaders/dataset_collision.py
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

COLORS = ['yellow', 'green', 'blue', 'red']  # one color per object type!
OBJECT_TYPE = ['sphere', 'cylinder_up', 'cylinder_down']
T = 15
MASSES = [1, 2, 5]
FRICTIONS = [0.1, 0.5, 1]
RESTITUTIONS = [0.1, 0.5, 1]


class Collision_CF(data.Dataset):
    """
    Dataset size (train/val/test):
      - normal: 14k/4k/2k
      - moving_sphere: 8k/1.8k/2k
      - moving_cylinder: 8k/2.1k/2k
    """

    def __init__(self,
                 type,
                 split,
                 root_dir='/usr/local/google/home/fbaradel/Dataset/CoPhy/collisionCF',
                 is_rgb=False,
                 only_cd=True,
                 pose_3d_fn='states.npy',
                 use_preextracted_object_properties=False,
                 preextracted_object_properties_dir='',
                 num_examples=None,
                 *args, **kwargs,
                 ):
        assert type in ['moving_cylinder', 'moving_sphere', 'normal']
        assert split in ['train', 'test', 'val']
        self.pose_3d_fn = pose_3d_fn
        self.only_cd = only_cd
        self.type = type
        self.split = split
        self.is_rgb = is_rgb
        self.is_training = True if split == 'train' else False
        self.root_dir = root_dir
        self.num_objects = len(COLORS)
        self.colors = COLORS
        self.use_preextracted_object_properties = use_preextracted_object_properties
        self.preextracted_object_properties_dir = preextracted_object_properties_dir

        # with open(os.path.join(self.root_dir, f"{self.split}_{self.type}.txt"), 'r') as f:
        with open(f"./dataloaders/splits/collisionCF_{self.split}_{self.type}.txt", 'r') as f:
            lines = f.readlines()
        self.list_ex = [x.strip() for x in lines]
        if num_examples:
            self.list_ex = self.list_ex[:num_examples]

        if self.use_preextracted_object_properties:
            fn = os.path.join(self.preextracted_object_properties_dir,
                              f"collision_{self.type}_{self.split}_extracted_prop.pickle")
            with open(fn, 'rb') as f:
                self.dict_id2object_properties = pickle.load(f)

        print(self.__repr__())

    def __len__(self):
        return len(self.list_ex)

    def __repr__(self):
        return f"CollisionCF: Type of moving object={self.type} - Split={self.split} - N={self.__len__()}"

    def __getitem__(self, item):
        """ Extract one example of the dataset """

        # Sequences
        ex = self.list_ex[item]
        cd = os.path.join(self.root_dir, ex, 'cd')
        ab = os.path.join(self.root_dir, ex, 'ab')

        # 3D - do not take into account the z coordinates
        pose_3D_cd, presence_cd = get_pose_3D(cd, fn=self.pose_3d_fn)  # (T, K, 3)
        pose_3D_ab, presence_ab = get_pose_3D(ab, fn=self.pose_3d_fn)
        stab_cd = get_stab(pose_3D_cd, presence_cd, t_delta=5, eps=0.05)  # (K)
        stab_ab = get_stab(pose_3D_ab, presence_ab, t_delta=5, eps=0.05)
        num_objects_ab = int(presence_ab.sum())
        num_objects_cd = int(presence_cd.sum())

        # 2D
        pose_2D_cd = get_pose_2D(cd)  # (T, K, 4)
        pose_2D_ab = get_pose_2D(ab)

        # Confounders
        confounders = get_confounders(cd)  # (K,mass-friction-restitution)
        masses, frictions, restitutions = get_onehot_confounders(confounders)

        # Object type
        obj_type_ab = get_obj_type(ab)
        obj_type_cd = get_obj_type(cd)

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
            'obj_type_ab': obj_type_ab,
            'obj_type_cd': obj_type_cd,
            'masses': masses,
            'frictions': frictions,
            'restitutions': restitutions,
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


def get_index(val, list_val):
    """ Because of a weird bug, sometimes 0.1 is in reality 0.0997877 """
    for i, x in enumerate(list_val):
        if x - val < 0.01:
            return i


def get_onehot_confounders(confounders):
    """ (4,3) matrix
    1 row per object and confounder type per column (mass-fristion-restitution)
    """
    masses = np.zeros((len(COLORS), len(MASSES)), dtype=np.float32)
    frictions = np.zeros((len(COLORS), len(FRICTIONS)), dtype=np.float32)
    restitutions = np.zeros((len(COLORS), len(RESTITUTIONS)), dtype=np.float32)
    for i in range(confounders.shape[0]):
        if np.abs(confounders[i]).sum() > 0:
            masses[i, get_index(confounders[i, 0], MASSES)] = 1.
            frictions[i, get_index(confounders[i, 1], FRICTIONS)] = 1.
            restitutions[i, get_index(confounders[i, 2], RESTITUTIONS)] = 1.
    return masses, frictions, restitutions


def get_obj_type(ex_dir):
    with open(os.path.join(ex_dir, 'colors.txt'), 'r') as f:
        lines = f.readlines()

    obj_type = np.zeros((len(COLORS), len(OBJECT_TYPE)), dtype=np.float32)
    for x in lines:
        _type = x.split('type=')[1].split(' ')[0]
        _col = x.split('color=')[1].strip()
        idx_type = OBJECT_TYPE.index(_type)
        idx_col = COLORS.index(_col)
        obj_type[idx_col, idx_type] = 1.

    return obj_type


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dataloader for CollisionCF.')
    parser.add_argument('--root_dir',
                        default='/usr/local/google/home/fbaradel/Dataset/CoPhy_224/collisionCF',
                        type=str,
                        help='Location of the data.')
    args = parser.parse_args()

    dataset = Collision_CF(type='normal',
                           root_dir=args.root_dir,
                           split='train',
                           is_rgb=True,
                           only_cd=False,
                           )

    # Drawings the bounding boxes on the video
    ex = dataset.__getitem__(1)
    add_bboxes_to_video(ex['rgb_ab'], ex['pose_2D_ab'], ex['presence_ab'], COLORS,
                        out_fn='./rgb_collision_ab_w_bboxes.mp4')
    add_bboxes_to_video(ex['rgb_cd'], ex['pose_2D_cd'], ex['presence_cd'], COLORS,
                        out_fn='./rgb_collision_cd_w_bboxes.mp4')
