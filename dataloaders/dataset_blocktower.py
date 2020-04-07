"""
BallCF dataloader

ipython dataloaders/dataset_blocktower.py
"""

import ipdb
from tqdm import tqdm
import os
import numpy as np
import argparse
import random
import torch.utils.data as data
from dataloaders.utils import *
import argparse
import pickle

COLORS = ['red', 'green', 'blue', 'yellow']
T = 30
MASSES = [1, 10]
FRICTIONS = [0.5, 1]
GRAVITY_X = [-0.5, 0, 0.5]
GRAVITY_Y = [-0.5, 0, 0.5]


class Blocktower_CF(data.Dataset):
    """
    Dataset size (train/val/test):
      - 3 blocks:
        - normal: 28.3k/8.0k/4.0k
        - generalization: 17.3k/3.0k/2k
      - 4 blocks:
        - normal: 23.4k/6.6k/3.3k
        - generalization: 14.5k/2.5k/2k
    """

    def __init__(self,
                 num_blocks,
                 split,
                 type,
                 root_dir='/usr/local/google/home/fbaradel/Dataset/CoPhy/blocktowerCF',
                 is_rgb=False,
                 only_cd=True,
                 pose_3d_fn='states.npy',
                 use_preextracted_object_properties=False,
                 preextracted_object_properties_dir='',
                 num_examples=None,
                 *args, **kwargs,
                 ):
        assert type in ['normal', 'generalization']
        assert split in ['train', 'test', 'val']
        self.pose_3d_fn = pose_3d_fn
        self.only_cd = only_cd
        self.num_blocks = num_blocks
        self.type = type
        self.split = split
        self.is_rgb = is_rgb
        self.is_training = True if split == 'train' else False
        self.root_dir = root_dir
        self.num_objects = len(COLORS)
        self.colors = COLORS
        self.use_preextracted_object_properties = use_preextracted_object_properties
        self.preextracted_object_properties_dir = preextracted_object_properties_dir

        if isinstance(self.num_blocks, int):
            with open(f"./dataloaders/splits/blocktowerCF_{self.split}_{self.num_blocks}_{self.type}.txt", 'r') as f:
                lines = f.readlines()
                self.list_ex = [x.strip() for x in lines]
        elif isinstance(self.num_blocks, list):
            assert len(self.num_blocks) == 2
            with open(f"./dataloaders/splits/blocktowerCF_{self.split}_{self.num_blocks[0]}_{self.type}.txt", 'r') as f:
                lines_1 = f.readlines()
                lines_1 = [x.strip() for x in lines_1]
            with open(f"./dataloaders/splits/blocktowerCF_{self.split}_{self.num_blocks[1]}_{self.type}.txt", 'r') as f:
                lines_2 = f.readlines()
                lines_2 = [x.strip() for x in lines_2]

            list_ex = lines_1 + lines_2
            list_num_blocks = [self.num_blocks[0] for x in lines_1] + [self.num_blocks[1] for x in lines_2]

            # shuffle idx
            idx = random.sample(list(range(len(list_ex))), len(list_ex))
            self.list_ex = [list_ex[x] for x in idx]
            self.list_num_blocks = [list_num_blocks[x] for x in idx]

        if num_examples:
            self.list_ex = self.list_ex[:num_examples]

        if self.use_preextracted_object_properties:
            fn = os.path.join(self.preextracted_object_properties_dir,
                              f"blocktower_{self.num_blocks}_{self.type}_{self.split}_extracted_prop.pickle")
            with open(fn, 'rb') as f:
                self.dict_id2object_properties = pickle.load(f)

        print(self.__repr__())

    def __len__(self):
        return len(self.list_ex)

    def __repr__(self):
        return f"BlocktowerCF: Number of balls={self.num_blocks} - Type={self.type} - Split={self.split} - N={self.__len__()}"

    def __getitem__(self, item):
        """ Extract one example of the dataset """

        # Sequences
        ex = self.list_ex[item]
        if isinstance(self.num_blocks, list):
            num_blocks = self.list_num_blocks[item]
        else:
            num_blocks = self.num_blocks

        cd = os.path.join(self.root_dir, str(num_blocks), ex, 'cd')
        ab = os.path.join(self.root_dir, str(num_blocks), ex, 'ab')

        # 3D - do not take into account the z coordinates
        pose_3D_cd, presence_cd = get_pose_3D(cd, fn=self.pose_3d_fn)  # (T, K, 3)
        pose_3D_ab, presence_ab = get_pose_3D(ab, fn=self.pose_3d_fn)
        stab_cd = get_stab(pose_3D_cd, presence_cd)  # (K)
        stab_ab = get_stab(pose_3D_ab, presence_ab)
        num_objects_ab = int(presence_ab.sum())
        num_objects_cd = int(presence_cd.sum())

        # 2D
        pose_2D_cd = get_pose_2D(cd)  # (T, K, 4)
        pose_2D_ab = get_pose_2D(ab)

        # Confounders
        confounders = get_confounders(cd)  # (4,mass+friction)
        with open(os.path.join(self.root_dir, str(num_blocks), ex, 'gravity.txt'), 'r') as f:
            lines = f.readline()
        gravity_x = float(lines.split('gravity_x=')[1].split(' ')[0])
        gravity_y = float(lines.split('gravity_y=')[1].strip())

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
            'gravity_x': gravity_x,
            'gravity_y': gravity_y,
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
    parser = argparse.ArgumentParser(description='Dataloader for BlocktowerCF.')
    parser.add_argument('--root_dir',
                        default='/storage/Datasets/CoPhy/CoPhy_224/blocktowerCF',
                        type=str,
                        help='Location of the data.')
    args = parser.parse_args()

    dataset = Blocktower_CF(num_blocks=3,
                            type='normal',
                            root_dir=args.root_dir,
                            split='train',
                            is_rgb=True,
                            only_cd=False)

    # Drawings the bounding boxes on the video
    ex = dataset.__getitem__(10)
    add_bboxes_to_video(ex['rgb_ab'], ex['pose_2D_ab'], ex['presence_ab'], COLORS,
                        out_fn='./rgb_blocktower_ab_w_bboxes.mp4')
    add_bboxes_to_video(ex['rgb_cd'], ex['pose_2D_cd'], ex['presence_cd'], COLORS,
                        out_fn='./rgb_blocktower_cd_w_bboxes.mp4')
