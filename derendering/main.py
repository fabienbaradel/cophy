"""
Training of the de-rendering module for each dataset.
Goal: given an image, the system is detecting the objects present in the scene and their properties

# debugging:
ipython derendering/main.py

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


def get_losses(gt_presence, gt_pose_3D, gt_pose_2D,
               pred_presence, pred_pose_3D, pred_pose_2D,
               w_presence=100., w_2d=1., w_3d=1.
               ):
    loss_presence = F.binary_cross_entropy_with_logits(pred_presence, gt_presence)  # []
    presence = (pred_presence > 0).float() * gt_presence
    if presence.sum().item() > 0:
        loss_2d = torch.sum(((pred_pose_2D - gt_pose_2D) ** 2).mean(-1) * presence) / torch.sum(presence)  # (B,K)
        loss_3d = torch.sum(((pred_pose_3D - gt_pose_3D) ** 2).mean(-1) * presence) / torch.sum(presence)  # (B,K)
        total_loss = w_presence * loss_presence + w_2d * loss_2d + w_3d * loss_3d
        return total_loss, (loss_presence, loss_2d, loss_3d)
    return w_presence * loss_presence, (loss_presence, loss_presence, loss_presence)


def get_acc_presence(pred, gt):
    acc = 1. - torch.mean(torch.abs((pred > 0).float() - gt))
    return acc


def get_mse(pred, gt, presence):
    mse = torch.sum(((pred - gt) ** 2).mean(-1) * presence) / torch.sum(presence)
    return mse


def get_iou(pred, gt, presence, resolution=224):
    boxA = gt * resolution
    boxB = pred * resolution

    # Resize [B,4,4] to [-1,4]
    boxA = boxA.view(-1, 4)
    boxB = boxB.view(-1, 4)
    presence_AB = presence.view(-1)

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = torch.max(boxA[:, 0], boxB[:, 0])
    yA = torch.max(boxA[:, 1], boxB[:, 1])
    xB = torch.min(boxA[:, 2], boxB[:, 2])
    yB = torch.min(boxA[:, 3], boxB[:, 3])

    # compute the area of intersection rectangle
    interArea = torch.clamp(xB - xA + 1, min=0) * torch.clamp(yB - yA + 1, min=0)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[:, 2] - boxA[:, 0] + 1) * (boxA[:, 3] - boxA[:, 1] + 1)
    boxBArea = (boxB[:, 2] - boxB[:, 0] + 1) * (boxB[:, 3] - boxB[:, 1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / (boxAArea + boxBArea - interArea)

    iou_mean = iou.sum() / presence_AB.sum()
    return iou_mean


def train_one_epoch(model, device, train_loader, optimizer,
                    log_file,
                    print_freq=50, w_presence=100., w_2d=1., w_3d=1.):
    model.train()

    end = time.time()
    list_acc_presence, list_iou_2d, list_mse_3d = [], [], []
    for i, input in enumerate(tqdm(train_loader)):
        # select at timestep and rgb, presence, pose_2D and pose_3D
        B, T, C, H, W = input['rgb_cd'].shape
        K = input['pose_3D_cd'].shape[2]
        rgb = input['rgb_cd'].view(B * T, C, H, W).to(device)
        gt_pose_3D = input['pose_3D_cd'].view(B * T, K, 3).to(device)
        gt_pose_2D = input['pose_2D_cd'].view(B * T, K, 4).to(device)
        gt_presence = torch.unsqueeze(input['presence_cd'], 1).repeat(1, T, 1).view(B * T, K).to(device)
        data_time = time.time() - end

        # forward
        pred_presence, pred_pose_3D, pred_pose_2D = model(rgb)

        # loss
        loss, _ = get_losses(gt_presence, gt_pose_3D, gt_pose_2D,
                             pred_presence, pred_pose_3D, pred_pose_2D,
                             w_presence, w_2d, w_3d)

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        end = time.time()

        if i % print_freq == 0:
            # metrics
            acc_presence = 100. * get_acc_presence(pred_presence, gt_presence)
            mse_3d = get_mse(pred_pose_3D, gt_pose_3D, gt_presence)
            iou_2d = get_iou(pred_pose_2D, gt_pose_2D, gt_presence)
            list_acc_presence.append(acc_presence.item())
            list_iou_2d.append(iou_2d.item())
            list_mse_3d.append(mse_3d.item())

            print(f"{i}/{len(train_loader)} "
                  f"Data = {data_time:.3f} "
                  f"Loss = {loss:.4f} "
                  f"Acc_presence = {acc_presence:.1f} "
                  f"IOU_2d = {iou_2d:.6f} "
                  f"MSE_3d = {mse_3d:.6f}"
                  )

    # append to log file
    with open(log_file, "a+") as f:
        f.write(f"Acc_presence={np.mean(list_acc_presence):.1f} "
                f"IOU_2d={np.mean(list_iou_2d):.6f} "
                f"MSE_3d={np.mean(list_mse_3d):.6f}\n")


def validate(model, device, val_loader, log_dir, log_file, print_freq=200):
    model.eval()

    end = time.time()
    list_acc_presence, list_iou_2d, list_mse_3d = [], [], []
    for i, input in enumerate(tqdm(val_loader)):
        # select at timestep and rgb, presence, pose_2D and pose_3D
        B, T, C, H, W = input['rgb_cd'].shape
        K = input['pose_3D_cd'].shape[2]
        rgb = input['rgb_cd'].view(B * T, C, H, W).to(device)
        gt_pose_3D = input['pose_3D_cd'].view(B * T, K, 3).to(device)
        gt_pose_2D = input['pose_2D_cd'].view(B * T, K, 4).to(device)
        gt_presence = torch.unsqueeze(input['presence_cd'], 1).repeat(1, T, 1).view(B * T, K).to(device)
        data_time = time.time() - end

        # forward
        pred_presence, pred_pose_3D, pred_pose_2D = model(rgb)
        end = time.time()

        # metrics
        acc_presence = 100. * get_acc_presence(pred_presence, gt_presence)
        iou_2d = get_iou(pred_pose_2D, gt_pose_2D, gt_presence)
        mse_3d = get_mse(pred_pose_3D, gt_pose_3D, gt_presence)
        list_acc_presence.append(acc_presence.item())
        list_iou_2d.append(iou_2d.item())
        list_mse_3d.append(mse_3d.item())

        if i % print_freq == 0:
            print(f"{i}/{len(val_loader)} "
                  f"Data = {data_time:.3f} "
                  f"Acc_presence = {np.mean(list_acc_presence):.1f} "
                  f"IOU_2d = {np.mean(list_iou_2d):.6f} "
                  f"MSE_3d = {np.mean(list_mse_3d):.6f}"
                  )

            # vizu
            j = 0
            rgb = rgb.view(B, T, C, H, W).detach().cpu().numpy()
            pred_pose_2D = pred_pose_2D.view(B, T, K, 4).detach().cpu().numpy()
            pred_presence = (pred_presence[:B] > 0).float().detach().cpu().numpy()
            add_bboxes_to_video(rgb[j], pred_pose_2D[j], pred_presence[j],
                                val_loader.dataset.colors,
                                out_fn=os.path.join(log_dir, f"./rgb_pred_bboxes_{i:04d}.mp4"))

    # append to log file
    with open(log_file, "a+") as f:
        f.write(f"Acc_presence={np.mean(list_acc_presence):.1f} "
                f"IOU_2d={np.mean(list_iou_2d):.6f} "
                f"MSE_3d={np.mean(list_mse_3d):.6f}\n")


def get_dataloaders(dataset_name, dataset_dir, kwargs_loader,
                    num_training_examples=5000,
                    num_validation_examples=1000):
    # choice of dataset
    if dataset_name == 'balls':
        train_dataset = Balls_CF(num_balls=4,
                                 root_dir=dataset_dir,
                                 split='train',
                                 num_examples=num_training_examples,
                                 is_rgb=True)
        val_dataset = Balls_CF(num_balls=4,
                               root_dir=dataset_dir,
                               split='val',
                               num_examples=num_validation_examples,
                               is_rgb=True)
    elif dataset_name == 'collision':
        train_dataset = Collision_CF(type='normal',
                                     root_dir=dataset_dir,
                                     split='train',
                                     num_examples=num_training_examples,
                                     is_rgb=True)
        val_dataset = Collision_CF(type='normal',
                                   root_dir=dataset_dir,
                                   num_examples=num_validation_examples,
                                   split='val',
                                   is_rgb=True)
    elif dataset_name == 'blocktower':
        train_dataset = Blocktower_CF(type='normal',
                                      num_blocks=[3, 4],
                                      root_dir=dataset_dir,
                                      split='train',
                                      num_examples=num_training_examples,
                                      is_rgb=True)
        val_dataset = Blocktower_CF(type='normal',
                                    num_blocks=[3, 4],
                                    root_dir=dataset_dir,
                                    split='val',
                                    num_examples=num_validation_examples,
                                    is_rgb=True)
    else:
        raise NameError('Unkown dataset name.')

    # loader
    train_loader = DataLoader(train_dataset, shuffle=True, **kwargs_loader)
    kwargs_loader_val = kwargs_loader.copy()
    kwargs_loader_val['batch_size'] = 1
    val_loader = DataLoader(val_dataset, **kwargs_loader_val)

    return train_loader, val_loader


def main(args):
    # kwargs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs_loader = {'batch_size': args.batch_size}
    if device.type == 'cuda':
        kwargs_loader.update({'num_workers': args.workers, 'pin_memory': True})

    # datasets and loaders
    train_loader, val_loader = get_dataloaders(args.dataset_name,
                                               args.dataset_dir,
                                               kwargs_loader)

    # model and optim
    model = DeRendering(num_objects=train_loader.dataset.num_objects).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # training
    os.makedirs(args.log_dir, exist_ok=True)
    log_file_val = os.path.join(args.log_dir, 'val.txt')
    log_file_train = os.path.join(args.log_dir, 'train.txt')
    w_presence, w_2d, w_3d = 1., 1., 1.
    for epoch in range(1, args.epochs):
        train_one_epoch(model, device, train_loader, optimizer, log_file_train,
                        w_presence=w_presence,
                        w_2d=w_2d,
                        w_3d=w_3d)
        log_dir_val = os.path.join(args.log_dir, 'vizu_val', f"{epoch:02d}")
        os.makedirs(log_dir_val, exist_ok=True)
        validate(model, device, val_loader, log_dir_val, log_file_val)
        torch.save(model.state_dict(), os.path.join(args.log_dir, 'model_state_dict.pt'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training of the derendering module.')
    parser.add_argument('--dataset_dir',
                        default='/tmp/CoPhy_224/ballsCF',
                        # default='/storage/Datasets/CoPhy/CoPhy_224/ballsCF',
                        # default='/storage/Datasets/CoPhy/CoPhy_224/collisionCF',
                        # default='/storage/Datasets/CoPhy/CoPhy_224/blocktowerCF',
                        type=str,
                        help='Location of the data.')
    parser.add_argument('--log_dir',
                        default='/tmp/cophy_derendering',
                        type=str,
                        help='Location of the log dir.')
    parser.add_argument('--dataset_name',
                        default='balls',
                        # default='collision',
                        # default='blocktower',
                        type=str,
                        help='Which dataset to take (balls, collision, blocktower).')
    parser.add_argument('--batch_size',
                        # default=4,
                        default=8,
                        type=int, help='Batch size.')
    parser.add_argument('--workers',
                        # default=0,
                        default=8,
                        type=int, help='Workers.')
    parser.add_argument('--epochs', default=20, type=int, help='Num epochs.')
    args = parser.parse_args()

    main(args)
