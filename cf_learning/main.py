"""
Counterfactual learning

# debugging:
ipython cf_learning/main.py

"""

from dataloaders.dataset_collision import Collision_CF
from dataloaders.dataset_blocktower import Blocktower_CF
from dataloaders.dataset_balls import Balls_CF
from cf_learning.model import CoPhyNet, CopyC
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


def get_losses(pred_pose_d, pred_stab_d, pred_presence_cd,
               gt_pose_d, gt_stab_d, gt_presence_cd,
               w_stab=1., w_pose=1.,
               ):
    loss_stab = F.binary_cross_entropy_with_logits(pred_stab_d, gt_stab_d)
    presence = pred_presence_cd * gt_presence_cd
    stab = (pred_stab_d > 0).float() * gt_stab_d
    T = stab.shape[1]
    binary = (1 - stab) * presence.unsqueeze(1).repeat(1, T, 1)
    loss_3d = torch.sum(((pred_pose_d - gt_pose_d) ** 2).mean(-1) * binary) / torch.sum(binary)  # (B,K)

    total_loss = w_stab * loss_stab + w_pose * loss_3d

    return total_loss, (loss_stab, loss_3d)


def get_acc_stab(pred, gt):
    acc = 1. - torch.mean(torch.abs((pred > 0).float() - gt))
    return acc


def train_one_epoch(model, device, loader, optimizer,
                    log_file,
                    print_freq=50, D=3,
                    is_rgb=False):
    model.train()

    end = time.time()
    list_acc_stab, list_mse_3d = [], []
    loader.dataset.is_rgb = False
    for i, input in enumerate(tqdm(loader)):
        data_time = time.time() - end
        if is_rgb:
            rgb_ab = input['rgb_ab'].to(device)
            rgb_c = input['rgb_cd'][:,:1].to(device)
            pred_pose_d, pred_presence_cd, pred_stab_d = model(rgb_ab, rgb_c)
        else:
            pred_presence_cd = input['pred_presence_cd'].to(device)
            pred_presence_ab = input['pred_presence_ab'].to(device)
            pred_pose_cd = input['pred_pose_3D_cd'][:, :1].to(device)
            pred_pose_ab = input['pred_pose_3D_ab'].to(device)
            pred_pose_d, pred_presence_cd, pred_stab_d = model(None, None,
                                                          pred_presence_ab,
                                                          pred_pose_ab,
                                                          pred_presence_cd,
                                                          pred_pose_cd,
                                                          )


        end = time.time()

        # gt
        gt_pose_cd = input['pose_3D_cd'].to(device)
        gt_pose_d = gt_pose_cd[:, 1:]
        gt_presence_cd = input['presence_cd'].to(device)
        gt_stab_d = input['stab_cd'][:, :-1].to(device)

        # loss
        loss, _ = get_losses(pred_pose_d, pred_stab_d, pred_presence_cd,
                             gt_pose_d, gt_stab_d, gt_presence_cd,
                             w_stab=1., w_pose=10.)

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        end = time.time()

        if i % print_freq == 0:
            # metrics
            mse_3d = get_mse(pred_pose_d, gt_pose_d, pred_presence_cd, D=D).mean()
            acc_stab = 100. * get_acc_stab(pred_stab_d, gt_stab_d)
            list_mse_3d.append(mse_3d.item())
            list_acc_stab.append(acc_stab.item())

            print(f"{i}/{len(loader)} "
                  f"Data = {data_time:.3f} "
                  f"Loss = {loss:.4f} "
                  f"Acc_stab = {np.mean(list_acc_stab):.2f} "
                  f"MSE_3d = {np.mean(list_mse_3d):.6f}"
                  )

    # append to log file
    with open(log_file, "a+") as f:
        f.write(f"Acc_presence={np.mean(list_acc_stab):.2f} "
                f"MSE_3d={np.mean(list_mse_3d):.6f}\n")


def get_mse(pred, gt, presence, D=3):
    T = pred.shape[1]
    dist = ((pred[:, :, :, :D] - gt[:, :, :, :D]) ** 2).mean(-1) * presence.unsqueeze(1)  # (B,T,K)
    mse = dist.sum((1, 2)) / (presence.sum(1) * T)
    return mse


def validate(model, device, loader, log_dir, log_file, print_freq=100, D=3, is_rgb=True):
    model.eval()

    end = time.time()
    list_mse_3d = []
    loader.dataset.is_rgb = is_rgb
    for i, input in enumerate(tqdm(loader)):
        data_time = time.time() - end

        # pred
        if is_rgb:
            # from RGB
            rgb_ab = input['rgb_ab'].to(device)
            rgb_c = input['rgb_cd'][:, :1].to(device)
            pred_pose_d, pred_presence_cd, stab_d = model(rgb_ab, rgb_c)
        else:
            #fro preextracted visual object properties
            pred_presence_cd = input['pred_presence_cd'].to(device)
            pred_presence_ab = input['pred_presence_ab'].to(device)
            pred_pose_cd = input['pred_pose_3D_cd'][:, :1].to(device)
            pred_pose_ab = input['pred_pose_3D_ab'].to(device)
            pred_pose_d, pred_presence_cd, stab_d = model(None, None,
                                                          pred_presence_ab,
                                                          pred_pose_ab,
                                                          pred_presence_cd,
                                                          pred_pose_cd,
                                                          )
        end = time.time()

        # gt
        gt_pose_cd = input['pose_3D_cd'].to(device)
        gt_pose_d = gt_pose_cd[:, 1:]

        # metrics
        mse_3d = get_mse(pred_pose_d, gt_pose_d, pred_presence_cd, D=D).mean()
        list_mse_3d.append(mse_3d.item())

        if i % print_freq == 0:
            print(f"{i}/{len(loader)} "
                  f"Data = {data_time:.3f} "
                  f"MSE_3d = {np.mean(list_mse_3d):.6f}"
                  )
    # append to log file
    to_write = f"MSE_3d={np.mean(list_mse_3d):.6f}\n"
    print(f"\n***Results: {to_write}***\n")
    with open(log_file, "a+") as f:
        f.write(to_write)


def get_dataloaders(dataset_name, dataset_dir, kwargs_loader, num_objects=3, type='normal',
                    preextracted_obj_vis_prop_dir='',
                    train_from_rgb=False,
                    evaluate_on_test_only=False):
    # choice of dataset
    if dataset_name == 'balls':
        if not evaluate_on_test_only:
            train_dataset = Balls_CF(num_balls=num_objects,
                                     root_dir=dataset_dir,
                                     split='train',
                                     is_rgb=train_from_rgb,
                                     only_cd=False,
                                     use_preextracted_object_properties=not train_from_rgb,
                                     preextracted_object_properties_dir=preextracted_obj_vis_prop_dir, #'/usr/local/google/home/fbaradel/Documents/extracted_object_properties/ballCF'
                                     )
            val_dataset = Balls_CF(num_balls=num_objects,
                                   root_dir=dataset_dir,
                                   split='val',
                                   is_rgb=train_from_rgb,
                                   only_cd=False,
                                   use_preextracted_object_properties=not train_from_rgb,
                                   preextracted_object_properties_dir=preextracted_obj_vis_prop_dir,
                                   )
        else:
            train_dataset, val_dataset = None, None
        test_dataset = Balls_CF(num_balls=num_objects,
                                root_dir=dataset_dir,
                                split='test',
                                is_rgb=True,
                                only_cd=False,
                                use_preextracted_object_properties=False,
                                preextracted_object_properties_dir=preextracted_obj_vis_prop_dir,
                                )
        D = 2
    elif dataset_name == 'collision':
        if not evaluate_on_test_only:
            train_dataset = Collision_CF(type=type,
                                         root_dir=dataset_dir,
                                         split='train',
                                         is_rgb=train_from_rgb,
                                         only_cd=False,
                                         use_preextracted_object_properties=not train_from_rgb,
                                         preextracted_object_properties_dir=preextracted_obj_vis_prop_dir,
                                         )
            val_dataset = Collision_CF(type=type,
                                       root_dir=dataset_dir,
                                       split='val',
                                       is_rgb=train_from_rgb,
                                       only_cd=False,
                                       use_preextracted_object_properties=not train_from_rgb,
                                       preextracted_object_properties_dir=preextracted_obj_vis_prop_dir,
                                       )
        else:
            train_dataset, val_dataset = None, None
        test_dataset = Collision_CF(type=type,
                                    root_dir=dataset_dir,
                                    split='test',
                                    is_rgb=True,
                                    only_cd=False,
                                    use_preextracted_object_properties=False,
                                    preextracted_object_properties_dir=preextracted_obj_vis_prop_dir,
                                    )
        D = 3
    elif dataset_name == 'blocktower':
        if not evaluate_on_test_only:
            train_dataset = Blocktower_CF(type=type,
                                          num_blocks=num_objects,
                                          root_dir=dataset_dir,
                                          split='train',
                                          is_rgb=train_from_rgb,
                                          only_cd=False,
                                          use_preextracted_object_properties=not train_from_rgb,
                                          preextracted_object_properties_dir=preextracted_obj_vis_prop_dir,
                                          )
            val_dataset = Blocktower_CF(type=type,
                                        num_blocks=num_objects,
                                        root_dir=dataset_dir,
                                        split='val',
                                        is_rgb=train_from_rgb,
                                        only_cd=False,
                                        use_preextracted_object_properties=not train_from_rgb,
                                        preextracted_object_properties_dir=preextracted_obj_vis_prop_dir,
                                        )
        else:
            train_dataset, val_dataset = None, None
        test_dataset = Blocktower_CF(type=type,
                                     num_blocks=num_objects,
                                     root_dir=dataset_dir,
                                     split='test',
                                     is_rgb=True,
                                     only_cd=False,
                                     use_preextracted_object_properties=False,
                                     preextracted_object_properties_dir=preextracted_obj_vis_prop_dir,
                                     )
        D = 3
    else:
        raise NameError('Unkown dataset name.')

    # loader
    if not evaluate_on_test_only:
        train_loader = DataLoader(train_dataset, shuffle=True, **kwargs_loader)
        kwargs_loader_val = kwargs_loader.copy()
        kwargs_loader_val['batch_size'] = 8
        val_loader = DataLoader(val_dataset, **kwargs_loader_val)
        test_loader = DataLoader(test_dataset, **kwargs_loader_val)
        return train_loader, val_loader, test_loader, D
    else:
        kwargs_loader_val = kwargs_loader.copy()
        kwargs_loader_val['batch_size'] = 8
        test_loader = DataLoader(test_dataset, **kwargs_loader_val)
        return None, None, test_loader, D


def get_trainable_params(model):
    """ get list of parameters to train of a network """
    trainable_params = []
    for name_c, child in model.named_children():
        for name_p, param in child.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)

    return trainable_params


def main(args):
    # kwargs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs_loader = {'batch_size': args.batch_size}
    if device.type == 'cuda':
        kwargs_loader.update({'num_workers': args.workers, 'pin_memory': True})

    # datasets and loaders
    train_loader, val_loader, test_loader, D = get_dataloaders(args.dataset_name,
                                                               args.dataset_dir,
                                                               kwargs_loader,
                                                               args.num_objects,
                                                               args.type,
                                                               preextracted_obj_vis_prop_dir=args.preextracted_obj_vis_prop_dir,
                                                               train_from_rgb=args.train_from_rgb,
                                                               evaluate_on_test_only=args.evaluate,
                                                               )

    # model
    dict_model_fn = {'copy_c': CopyC, 'cophynet': CoPhyNet}
    model_fn = dict_model_fn[args.model]
    model = model_fn(num_objects=test_loader.dataset.num_objects).to(device)

    # load the derendering module
    pretrained_dict = torch.load(args.derendering_ckpt)
    pretrained_dict = {'derendering.' + k: v for k, v in pretrained_dict.items()}
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model.load_state_dict(pretrained_dict, strict=False)

    # freeze derendering
    for param in model.derendering.parameters():
        param.requires_grad = False

    # optim and training
    if len(get_trainable_params(model)) > 0 and not args.evaluate:
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # training
        os.makedirs(args.log_dir, exist_ok=True)
        log_file_val = os.path.join(args.log_dir, 'val.txt')
        log_file_train = os.path.join(args.log_dir, 'train.txt')
        for epoch in range(1, args.epochs):
            train_one_epoch(model, device, train_loader, optimizer, log_file_train, D=D)
            log_dir_val = os.path.join(args.log_dir, 'vizu_val', f"{epoch:02d}")
            os.makedirs(log_dir_val, exist_ok=True)
            validate(model, device, val_loader, log_dir_val, log_file_val)
            torch.save(model.state_dict(), os.path.join(args.log_dir, 'model_state_dict.pt'))

    # testing
    if len(get_trainable_params(model)) > 0 and args.evaluate:
        # load ckpt
        model.load_state_dict(torch.load(args.pretrained_ckpt), strict=True)
    os.makedirs(args.log_dir, exist_ok=True)
    log_file_test = os.path.join(args.log_dir, 'test.txt')
    log_dir_test = os.path.join(args.log_dir, 'vizu_test', f"00")
    validate(model, device, test_loader, log_dir_test, log_file_test, D=D, is_rgb=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training of the derendering module.')
    parser.add_argument('--dataset_dir',
                        # default='/usr/local/google/home/fbaradel/Dataset/CoPhy_224/ballsCF',
                        # default='/usr/local/google/home/fbaradel/Dataset/CoPhy_224/collisionCF',
                        default='/usr/local/google/home/fbaradel/Dataset/CoPhy_224/blocktowerCF',
                        type=str,
                        help='Location of the data.')
    parser.add_argument('--num_objects',
                        default=3,
                        type=int,
                        help='Number of objects for training.')
    parser.add_argument('--type',
                        default='normal',
                        type=str,
                        help='Type of train/val/test split.')
    parser.add_argument('--derendering_ckpt',
                        # default='/usr/local/google/home/fbaradel/Documents/log_dir/ballsCF/model_state_dict.pt',
                        default='/usr/local/google/home/fbaradel/Documents/log_dir/blocktowerCF/model_state_dict.pt',
                        type=str,
                        help='Location of the pre-trained derendering module.')
    parser.add_argument('--log_dir',
                        default='/tmp/cophy_cf_learning',
                        type=str,
                        help='Location of the log dir.')
    parser.add_argument('--dataset_name',
                        # default='balls',
                        # default='collision',
                        default='blocktower',
                        type=str,
                        help='Which dataset to take (balls, collision, blocktower).')
    parser.add_argument('--model',
                        # default='copy_c',
                        # default='copy_b',
                        default='cophynet',
                        type=str,
                        help='Model name to use.')
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size.')
    parser.add_argument('--workers', default=8, type=int, help='Workers.')
    parser.add_argument('--epochs', default=20, type=int, help='Num epochs.')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true')
    parser.add_argument('--no-evaluate', dest='evaluate', action='store_false')
    parser.set_defaults(evaluate=False)
    parser.add_argument('--train-from-rgb', dest='train_from_rgb', action='store_true')
    parser.add_argument('--no-train-from-rgb', dest='train_from_rgb', action='store_false')
    parser.set_defaults(train_from_rgb=False)
    parser.add_argument('--pretrained_ckpt',
                        default='/usr/local/google/home/fbaradel/Documents/log_dir/blocktowerCF/model_state_dict.pt',
                        type=str,
                        help='Location of the pre-trained derendering module.')
    parser.add_argument('--preextracted_obj_vis_prop_dir',
                        default='',
                        type=str,
                        help='Location of the pre-extracted object visual properties.')

    args = parser.parse_args()

    main(args)
