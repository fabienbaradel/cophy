# CoPhy: Counterfactual Learning of Physical Dynamics

This repository contains the dataloaders for the benchmark introduced in ["CoPhy: Counterfactual Learning of Physical Dynamics", F. Baradel, N. Neverova, J. Mille, G. Mori, C. Wolf, ICLR'2020](https://arxiv.org/abs/1909.12000).

Links: [Project page](https://projet.liris.cnrs.fr/cophy/) | [Data](https://zenodo.org/record/3674790#.XotcTtMza-o) | [Video](https://youtu.be/HHbBJK6F8nE)

<img src="teaser.png" width="800"/>

## Dataset
You first need to download the dataset (18Go) which is hosted on Zenodo by clicking on this [link](https://zenodo.org/record/3674790/files/cophy_224.tar.gz?download=1).
Then you should unzip the file and you should get data stored like that:
```
CoPhy_224/
├── ballsCF
│   ├── 2
│   ├── 3
│   ├── 4
│   ├── 5
│   └── 6
├── blocktowerCF
│   ├── 3
│   └── 4
└── collisionCF
    ├── 1
    ├── ...
    └── 9999
```

Below are some explanations for the files associated to each dataset:
* BlocktowerCF
    ```
    CoPhy_224/blocktowerCF/<nb-blocks>/<id> # number of blocks composing the blocktower and the example id. 
    ├── ab # the observed sequence (A,B)
    │   ├── bboxes.npy # bounding box pixel coordinates of each block
    │   ├── colors.txt # color of each block from bottom to top
    │   ├── rgb.mp4 # the RGB sequence of 6 seconds long at fps=5 (30 timesteps)
    │   ├── segm.mp4 # the associated segmentation fo each block
    │   └── states.npy # the sequence of state information of each block (3D position, 4D quaternion and their associated velocities)
    ├── cd # sequence including the modified initial state C and the outcome D
    │   ├── bboxes.npy
    │   ├── colors.txt
    │   ├── rgb.mp4
    │   ├── segm.mp4
    │   └── states.npy
    ├── confounders.npy # the confounder information for each block
    └── gravity.txt # the gravity of the scene (also a confounder for this dataset) 
    ```
* BallsCF
    ```
        CoPhy_224/ballsCF/<nb-balls>/<id> # number of balls in the scene and the example id
    ├── ab
    │   ├── bboxes.npy
    │   ├── colors.txt
    │   ├── rgb.mp4
    │   ├── segm.mp4
    │   └── states.npy
    ├── cd
    │   ├── bboxes.npy
    │   ├── colors.txt
    │   ├── rgb.mp4
    │   ├── segm.mp4
    │   └── states.npy
    ├── confounders.npy
    └── explanations.txt # explanations about the do-intervention
    ```
* CollisionCF
    ```
    CoPhy_224/collisionCF/<id> # the example id
    ├── ab
    │   ├── bboxes.npy
    │   ├── colors.txt
    │   ├── rgb.mp4
    │   ├── segm.mp4
    │   └── states.npy
    ├── cd
    │   ├── bboxes.npy
    │   ├── colors.txt
    │   ├── rgb.mp4
    │   ├── segm.mp4
    │   └── states.npy
    ├── confounders.npy
    └── explanations.txt
    ```

You can find the train/val/test splits [here](https://zenodo.org/record/3674790/files/splits.zip?download=1) or directly on the subdirectory ```dataloaders/splits``` of this repo.

By running the python script ```bias_free_blocktower.py```, you can see that our datasets and in particularly BlocktowerCF is close to bias-free w.r.t. the overall stability of the full blocktwer.
```
COPHY=<ROOT-LOCATION-OF-COPHY>
python bias_free_blocktower.py --dir $COPHY/blocktowerCF/3 # for blocktower composed of 3 blocks
python bias_free_blocktower.py --dir $COPHY/blocktowerCF/4 # and for blocktower composed of 4 blocks
```

## Dataloaders
We provide the dataloaders for each dataset in the directory ```dataloaders ``` in this repo:
```
dataloaders/
├── dataset_balls.py # dataloader for BallsCF
├── dataset_blocktower.py # dataloader for BlocktowerCF
├── dataset_collision.py # dataloader for CollisionCF
├── splits
│   ├── ballsCF_test_2.txt # id list of examples composing the test set for ballsCF
│   ├── ...
│   └── collisionCF_val_normal.txt
└── utils.py # utils functions for load videos, extracting 3D positions etc..;
```

First let's make sure that you are able to iterate the elements of each dataset.
By running the following command line:
```
COPHY=<root-location-of-cophy-benchmark>
python dataloaders/dataset_blocktower.py --root_dir $COPHY/blocktowerCF
```
You should find two mp4 files in your directory that have been just saved: ```rgb_<dataset>_ab_w_bboxes.mp4``` and ```rgb_<dataset>_ab_w_bboxes.mp4```.
It corresponds to a (A,B) and (C,D) RGB sequence with the bonding box visualization around each objects.
You can do the same for the two other datasets.

## Training
### Training the derendering
The first step is to train the derendering modules for each dataset.
This can be done by running the following script:
```
COPHY=/tmp/CoPhy_224
LOGDIR=/tmp/logdir/derendering
./derendering/train_derendering.sh $COPHY $LOGDIR
```
The pre-trained checkpoints for each dataset are located in the repository in the subdirectory ```ckpts/derendering/```

For speeding up the training procedure for the counterfactual learning part, we can extract the 3D position of each object using the derendering modules.
This can be done using the following command line:
```
DERENDERING_DIR=./ckpts/derendering
OUT_DIR=/tmp/extracted_obj_visu_prop
./derendering/extract_3d_pose.sh $COPHY $DERENDERING_DIR $OUT_DIR
```
This is extracting the object presence, 3D pose and bounding box locations for each dataset splits.

### Copying baseline
Scripts for evaluating the copying baseine ('Copy C') on all test sets:
```
LOG_DIR=/tmp/log_dir/copy_c
COPHY=/storage/Datasets/CoPhy/CoPhy_224
DERENDERING=./ckpts/derendering
./cf_learning/run_copying_baselines.sh $LOG_DIR $COPHY $DERENDERING
```

### Training CoPhyNet from estimated poses
Script for training on all datasets as well as testing for all train/test splits.
```
LOG_DIR=/tmp/log_dir/cophynet
COPHY=/storage/Datasets/CoPhy/CoPhy_224
DERENDERING=./ckpts/derendering
PREXTRACTED_OBJ=/storage/Datasets/CoPhy/extracted_object_properties
./cf_learning/train_cophynet.sh $LOG_DIR $COPHY $DERENDERING $PREXTRACTED_OBJ
```

### Evaluation from pre-trained models
TODO

## Citation
If you find this paper or the benchmark useful for your research, please cite our paper.
```
@InProceedings{Baradel_2020_ICLR,
author = {Baradel, Fabien and Neverova, Natalia and Mille, Julien and Mori, Greg and Wolf, Christian},
title = {CoPhy: Counterfactual Learning of Physical Dynamics},
booktitle = {ICLR},
year = {2020}
}
```

## Acknowledgements
This work was funded by grant Deepvision (ANR-15- CE23-0029, STPGP-479356-15), a joint French/Canadian call by ANR & NSERC.

