# CoPhy: Counterfactual Learning of Physical Dynamics

This repository contains the code associated to the paper ["CoPhy: Counterfactual Learning of Physical Dynamics", F. Baradel, N. Neverova, J. Mille, G. Mori, C. Wolf, ICLR'2020](https://arxiv.org/abs/1909.12000).

Links: [Project page](https://projet.liris.cnrs.fr/cophy/) | [Data](https://zenodo.org/record/3674790#.XotcTtMza-o) | [Video](https://youtu.be/HHbBJK6F8nE)

<p align="center">

<img src="cophy.gif" width="300" height="300" />

<img src="teaser.png" width="800"/>

</p>

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

By running the python script ```bias_free_blocktower.py```, you can see that the BlocktowerCF dataset is close to bias-free w.r.t. the overall stability of the full blocktwer.
```
COPHY=<ROOT-LOCATION-OF-COPHY>
python bias_free_blocktower.py --dir $COPHY/blocktowerCF/3 # for blocktower composed of 3 blocks
python bias_free_blocktower.py --dir $COPHY/blocktowerCF/4 # and for blocktower composed of 4 blocks
```

## Requirements
The code was tested 16.04 with Anaconda under Python 3.6 and Pytorch-1.4.
One GPU is required for training and testing.

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
The first step consist of training the derendering module for each dataset.
This can be done by running the following script:
```
DERENDERING_DIR=/tmp/logdir/derendering
./derendering/train_derendering.sh $COPHY $LOGDIR
```
If you don't want to train the derendering module, you can find pre-trained checkpoints for each dataset are located in the repository in the subdirectory ```ckpts/derendering/```

For speeding up the training procedure for the counterfactual learning part, we can extract the 3D position of each object using the derendering modules.
This can be done using the following command line:
```
DERENDERING_DIR=./ckpts/derendering
PREXTRACTED_OBJ=/tmp/extracted_obj_visu_prop
./derendering/extract_3d_pose.sh $COPHY $DERENDERING_DIR $OUT_DIR
```
This is extracting the object presence, 3D pose and bounding box locations for each dataset split.

### Copying baseline
Before training CoPhyNet, you can have a look at naive baselines such as copying the present in the future (denoted 'Copy C').
Running the evaluation on all datasets and all splits can be done using the following script:
```
LOG_DIR=/tmp/log_dir/copy_c
./cf_learning/run_copying_baselines.sh $LOG_DIR $COPHY $DERENDERING
```

### Training CoPhyNet from estimated poses
Now, you can train CoPhyNet and evaluating its performance for each train/val/test split.
This can be done using the following command:
```
LOG_DIR=/tmp/log_dir/cophynet
./cf_learning/train_cophynet.sh $LOG_DIR $COPHY $DERENDERING_DIR $PREXTRACTED_OBJ
```
You can find a file ```test.txt``` in each log directory which indicates the final performance on the test set.

### Evaluation using pre-trained models
We have also released some pre-trained models.
For example for the BlocktwerCF dataset, using the command below you can evaluate the model trained on blocktower composed of 3 blocks under normal settings on the test set which contains only towers composed of 4 blocks.
```
LOGDIR=/tmp/
python cf_learning/main.py \
--dataset_dir $COPHY/blocktowerCF
--pretrained_ckpt ./ckpts/cophynet/blocktowerCF/3/normal/model_state_dict.pt
--log_dir $LOGDIR/blocktowerCF/3/normal_4
--dataset_name blocktower
--model cophynet
--num_objects 4
--type normal
--evaluate
```
You should reach a performance around 0.480.
You can find the pre-trained models under different settings (3/4 balls, normal/generalization confounder distribution) in the subdirectory ```ckpts/cophynet/cophynet```.


One more example for the Balls dataset.
Run the following command for evaluating on the test set composed of 2 moving balls only, the model trained on 4 balls.
```
LOGDIR=/tmp/
python cf_learning/main.py \
--dataset_dir $COPHY/ballsCF \
--pretrained_ckpt ./ckpts/cophynet/ballsCF/4/model_state_dict.pt \
--log_dir $LOGDIR/ballsCF/4_2 \
--dataset_name balls \
--model cophynet \
--num_objects 5 \
--evaluate
```

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

