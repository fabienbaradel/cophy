import ipdb
from tqdm import tqdm
import os
import numpy as np
import random
import torch.utils.data as data
import imageio
from PIL import Image, ImageDraw
import time


def get_pose_3D(subdir, fn='states.npy'):
  """ Extract the sequence of 3D pose for each object as well as the binary presence """
  # Load
  fn = os.path.join(subdir, fn)
  states = np.load(fn) # (T,K,D)

  # Extract the pose only
  pose = states[:,:,:3] # (T,K,D) -> (number-of-timesteps,number-of-objects,3)

  # Presence
  presence = np.sum(np.abs(states[0,:,:3]), axis=-1) # we set the pose to (0,0,0) if the object was not present
  presence = presence > 0

  return pose.astype(np.float32), presence.astype(np.float32)


def get_pose_2D(subdir, resolution=224.):
  """ Extract the sequence of 2D pose for each object (pixel space)
  Bboxes are of type (x1,y1,x2,y2) in the pixel space (image of size 100x100).
  """
  # Load
  fn = os.path.join(subdir, 'bboxes.npy')
  bboxes = np.load(fn)  # (T,K,4)

  # try:
  #   bboxes = np.load(fn) # (20,9,4)
  # except:
  #   bboxes = np.zeros((30,4,4))

  bboxes /= resolution

  return bboxes.astype(np.float32)

def get_stab(pose, presence, t_delta=2, eps=0.05):
  """ Estimate the groud-truth stability per timesteps """
  stab = np.zeros_like(pose[:,:,0])
  T = pose.shape[0]
  for t in range(T):
    # Pose before and after
    t_before = t - t_delta
    t_before = 0 if t_before < 0 else t_before
    t_after = t + t_delta
    t_after = T-1 if t_after >= T else t_after

    # Delta to observed if the object has moved given an epsilon
    delta_pose = np.abs(pose[t_before] - pose[t_after])
    delta_pose = np.sum(delta_pose, axis=1)
    stab_t = delta_pose > eps
    stab[t] = stab_t.astype(np.float32)

  return stab.astype(np.float32) * presence # TODO make sure that it works

def get_confounders(subdir):
  """ Extract the confounders for each objects  """
  subdir = subdir.replace('cd', '')
  fn = os.path.join(subdir, 'confounders.npy')
  confounders = np.load(fn) # (K,H)
  return confounders.astype(np.float32).round(1)

def get_rgb(subdir):
  """
  Load the sequence of RGB frames from a mp4 file

  Requirement:
  pip install imageio-ffmpeg
  """

  fn = os.path.join(subdir, 'rgb.mp4')
  reader = imageio.get_reader(fn, pixelformat='yuvj444p')
  list_im = []
  # start = time.time()
  for i, im in enumerate(reader):
    list_im.append(im) # (224,224,3) uint8
  video = np.stack(list_im, axis=0) # (T,224,224,3)
  video = video.astype(np.float32)
  video /=255. # (T,224,224,3) float32 values btw 0 and 1

  video = np.transpose(video, (0,3,1,2))

  video = video * np.asarray([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)  # std
  video = video + np.asarray([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)  # mean

  # print(time.time() - start)

  return video.astype(np.float32)


def show_img(np_array_uint8, out_fn):
  if len(np_array_uint8.shape) == 3:
    img = Image.fromarray(np_array_uint8, 'RGB')
  elif len(np_array_uint8.shape) == 2:
    img = Image.fromarray(np_array_uint8)
  else:
    raise NameError('Unknown data type to show.')

  img.save(out_fn)
  img.show()


def add_bboxes_to_img(rgb_array, np_bbox, presence, list_colors, margin=1):
  """ Show the bounding box on a RGB image
  rgb_array: a np.array of shape (H,W,3) - uint8 - range=[0,255]
  np_bbox: np.array of shape (9,4) and a bbox is of type [x1,y1,x2,y2]
  list_colors: list of string of length 9
  """
  assert np_bbox.shape[0] == len(list_colors)

  H, W, _ = rgb_array.shape
  img_rgb = Image.fromarray(rgb_array, 'RGB')
  draw = ImageDraw.Draw(img_rgb)
  N = np_bbox.shape[0]
  for i in range(N):
    if presence[i] == 1:
      color = list_colors[i]
      x_1, y_1, x_2, y_2 = np_bbox[i]
      draw.rectangle(((H*x_1-margin, W*y_1-margin),
                      (H*x_2+margin, W*y_2+margin)),
                     outline=color, fill=None)
  return img_rgb

def add_bboxes_to_video(rgb, bboxes, presence, list_colors, out_fn):
  # reshape and rescale rgb
  rgb = rgb - np.asarray([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)  # mean
  rgb = rgb / np.asarray([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)  # std
  rgb *= 255.
  rgb = np.transpose(rgb, (0,2,3,1))
  rgb = rgb.astype(np.uint8)

  list_rgb_bboxes = []
  for t in range(rgb.shape[0]):
    list_rgb_bboxes.append(np.asarray(add_bboxes_to_img(rgb[t],
                                                        bboxes[t],
                                                        presence,
                                                        list_colors)))
  rgb_ab_w_bboxes = np.stack(list_rgb_bboxes)
  imageio.mimwrite(out_fn, rgb_ab_w_bboxes , fps = 5.,
                   quality=10, pixelformat='yuvj444p',
                   ffmpeg_log_level='quiet')