import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from derendering.model import DeRendering
import ipdb


class CopyC(nn.Module):
  def __init__(self, num_objects=5):
    super().__init__()

    self.derendering = DeRendering(num_objects)

  def forward(self, rgb_ab, rgb_c):

    # derender
    presence_ab, presence_c, \
    pose_3d_ab, pose_3d_c = extract_pose_ab_c(self.derendering, rgb_ab, rgb_c)
    T = rgb_ab.shape[1]

    # copy c T times
    pose_3d_d = pose_3d_c.repeat(1,T-1,1,1)

    return pose_3d_d, presence_c, None

def aggreg_E(E, presence):
  list_e = []
  K = E.size(1)
  presence = presence.unsqueeze(-1)
  presence_1 = presence.unsqueeze(1).repeat(1, K, 1, 1)  # (B, K, K, D)
  presence_2 = presence.unsqueeze(2).repeat(1, 1, K, 1)  # (B, K, K, D)
  presence_12 = presence_1 * presence_2
  for k in range(K):
    other_k = [x for x in range(K) if x != k]
    e_k = (E[:,k,other_k] * presence_12[:,k,other_k]).sum(1) # (B,H)
    e_k = e_k /  (0.01+presence_12[:,k,other_k].sum(1))
    list_e.append(e_k)
  E = torch.stack(list_e, 1) # (B,K,H)

  # E_sum
  E_sum = (E * presence).sum(1) / (0.01 + presence.sum(1)) # (B,H)
  E_sum = E_sum.unsqueeze(1).repeat(1,K,1)

  return E, E_sum

class CoPhyNet(nn.Module):
  def __init__(self, num_objects=5):
    super().__init__()

    # CNN
    self.derendering = DeRendering(num_objects)
    self.K = num_objects

    # AB
    H = 32
    self.mlp_inter = nn.Sequential(nn.Linear(2*(3),H),
                                   nn.ReLU(),
                                   nn.Linear(H,H),
                                   nn.ReLU(),
                                   nn.Linear(H,H),
                                   nn.ReLU(),
                                   )
    D = H
    # D += 3 if num_objects == 4 else 0 # TODO collisionCF only
    self.D = D
    self.mlp_out = nn.Sequential(nn.Linear(3+H+H, H),
                                 nn.ReLU(),
                                 nn.Linear(H, H))

    # RNN
    self.rnn = nn.GRU(D,H, num_layers=1, batch_first=True)

    # Stability
    self.mlp_inter_stab = nn.Sequential(nn.Linear(2*(H+3),H),
                                        nn.ReLU(),
                                        nn.Linear(H,H),
                                        nn.ReLU(),
                                        nn.Linear(H,H),
                                        nn.ReLU(),
                                        )
    self.mlp_stab = nn.Sequential(nn.Linear(H+H+H+3, H),
                                  nn.ReLU(),
                                  nn.Linear(H, 1))

    # Next position
    self.mlp_inter_delta = nn.Sequential(nn.Linear(2*(H+3),H),
                                         nn.ReLU(),
                                         nn.Linear(H,H),
                                         nn.ReLU(),
                                         nn.Linear(H,H),
                                         nn.ReLU(),
                                         )
    self.mlp_gcn_delta = nn.Sequential(nn.Linear(H*3 + 3, H),
                                       nn.ReLU(),
                                       nn.Linear(H, H))
    self.rnn_delta = nn.GRU(H,H, num_layers=1, batch_first=True)
    self.fc_delta = nn.Linear(H, 3)


    # args
    self.iterative_stab = True


  def gcn_on_AB(self, pose_ab, presence_ab):
    list_out = []
    K = pose_ab.size(2)
    T = pose_ab.size(1)
    for i in range(T):
      x = pose_ab[:,i,:,:3] # (B,4,3)

      # compute interactions : e_t+1 = f(o_t^1,o_t^2,r)
      x_1 = x.unsqueeze(1).repeat(1, K, 1, 1)  # (B, K, K, D)
      x_2 = x.unsqueeze(2).repeat(1, 1, K, 1)  # (B, K, K, D)
      x_12 = torch.cat([x_1, x_2], -1)
      E = self.mlp_inter(x_12) # B,K,K,H
      E, E_sum = aggreg_E(E, presence_ab) # B,K,H

      # next position : o_t+1^1 = f(o_t^1,e_t+1)
      out = self.mlp_out(torch.cat([x, E, E_sum], -1))
      list_out.append(out)

    out = torch.stack(list_out, 1) # (B,T,K,H)

    return out # (B,T,K,3)

  def rnn_on_AB_up(self, seq_o, object_type=None):
    if object_type is not None:
      T = seq_o.shape[1]
      object_type = object_type.unsqueeze(2).repeat(1,1,T,1)

    K = seq_o.size(2)
    list_out = []
    for k in range(K):
      x = seq_o[:,:,k]

      if object_type is not None:
        x = torch.cat([x, object_type[:,k]], -1)

      out, _ = self.rnn(x) # (B,T,H)
      list_out.append(out[:,-1])

    out = torch.stack(list_out, 1) # (B,K,H)
    return out

  def pred_stab(self, confounders, pose_t, presence):
    """
    Given a timestep - predict the stability per object
    :param confounders: (B,K,D)
    :param pose_t: (B,K,3)
    :param presence: (B,K)
    :return: stab=(B,K,1)
    """
    list_stab = []
    # x = input['pose_cd'][:,0] # (B,4,3)
    x = pose_t # (B,4,3)
    x = torch.cat([confounders, x], -1)
    K = x.size(1)

    # compute interactions : e_t+1 = f(o_t^1,o_t^2,r)
    x_1 = x.unsqueeze(1).repeat(1, K, 1, 1)  # (B, K, K, D)
    x_2 = x.unsqueeze(2).repeat(1, 1, K, 1)  # (B, K, K, D)
    x_12 = torch.cat([x_1, x_2], -1)
    E = self.mlp_inter_stab(x_12) # B,K,K,H
    # E, E_sum = aggreg_E(E, input['presence_cd']) # B,K,H
    E, E_sum = aggreg_E(E, presence) # B,K,H

    # stability
    stab = self.mlp_stab(torch.cat([x, E, E_sum], -1))

    return stab # (B,K,1)

  def pred_D(self, confounders, pose_3d_c, presence_c, T=30):
    """

    :param stability: (B,K,1)
    :param confounders: (B,K,D)
    :param input: pose_cd=(B,T,K,3) presence_cd=(B,K)
    :return: out=(B,10,K,3) stability=(B,10,4,1)
    """
    list_pose = []
    list_stability = []
    pose = pose_3d_c # (B,4,3)
    K = pose.size(1)
    list_last_hidden = []
    for i in range(T):
      # Stability prediction
      if i == 0 or self.iterative_stab == 'true':
        stability = self.pred_stab(confounders, pose, presence_c)
      list_stability.append(stability)

      # Cat
      x = torch.cat([pose, confounders], -1) # # .detach() ???

      # compute interactions : e_t+1 = f(o_t^1,o_t^2,r)
      x_1 = x.unsqueeze(1).repeat(1, K, 1, 1)  # (B, K, K, D)
      x_2 = x.unsqueeze(2).repeat(1, 1, K, 1)  # (B, K, K, D)
      x_12 = torch.cat([x_1, x_2], -1)
      E = self.mlp_inter_delta(x_12) # B,K,K,H
      E, E_sum = aggreg_E(E, presence_c) # B,K,H

      # next position : o_t+1^1 = f(o_t^1,e_t+1) with RNN on top
      _in = self.mlp_gcn_delta(torch.cat([x, E, E_sum], -1)) # (B,K,H)
      B = _in.size(0)
      list_new_hidden = []
      for k in range(K):
        if i == 0:
          hidden, *_ = self.rnn_delta(_in[:,[k]]) # (B,1,H)
        else:
          hidden, *_ = self.rnn_delta(_in[:,[k]], list_last_hidden[k].reshape(1,B,-1)) # (B,1,H)
        list_new_hidden.append(hidden)
      list_last_hidden = list_new_hidden
      hidden = torch.cat(list_last_hidden, 1) # (B,K,H)

      delta = self.fc_delta(hidden)

      if self.training:
        alpha = 0.01
        delta = delta * (1 - torch.sigmoid(stability/alpha)) # .detach() ???
      else:
        delta = delta * (1-(stability > 0).float())
      pose = pose + delta

      list_pose.append(pose)

    pose = torch.stack(list_pose, 1) # (B,T,K,3)
    stability = torch.stack(list_stability, 1) # (B,T,K,3)

    return pose, stability

  def forward(self, rgb_ab, rgb_c,
    pred_presence_ab, pred_pose_3d_ab,
    pred_presence_c, pred_pose_3d_c,
    pred_obj_type_ab=None, pred_obj_type_c=None,
  ):

    if rgb_ab is not None and rgb_c is not None:
      # derender
      presence_ab, presence_c, \
      pose_3d_ab, pose_3d_c = extract_pose_ab_c(self.derendering, rgb_ab, rgb_c)
    else:
      # already precomputed
      presence_ab = pred_presence_ab
      presence_c = pred_presence_c
      pose_3d_ab = pred_pose_3d_ab
      pose_3d_c = pred_pose_3d_c
      obj_type_ab, obj_type_c = pred_obj_type_ab, pred_obj_type_c

    # squeeze
    pose_3d_c = pose_3d_c[:,0] # (B,K,3)
    T = pose_3d_ab.shape[1] - 1

    # Run a GCN on AB
    seq_o = self.gcn_on_AB(pose_3d_ab, presence_ab) # (B,T,K,H)

    # Run a RNN on the outputs of GCN
    confounders = self.rnn_on_AB_up(seq_o, obj_type_ab) # (B,K,H)

    # pred
    out, stability = self.pred_D(confounders, pose_3d_c, presence_c, T=T)
    # stability = (B,T-1,K,1)
    stability = stability.squeeze(-1)

    return out, presence_c, stability


def extract_pose_ab_c(derendering, rgb_ab, rgb_c):
  rgb = torch.cat([rgb_ab, rgb_c], 1)
  B, T, H, W, C = rgb.shape
  rgb = rgb.view(B*T, H, W, C)
  presence, pose_3d, pose_2d = derendering(rgb)
  presence = presence.view(B, T, derendering.num_objects)
  presence = (presence > 0).float()
  presence_ab, presence_c = presence[:, 0], presence[:, -1] # TODO maybe avg over time for AB
  pose_3d = pose_3d.view(B, T, derendering.num_objects, 3)
  pose_3d_ab, pose_3d_c = pose_3d[:, :-1], pose_3d[:, -1:]

  return presence_ab, presence_c, pose_3d_ab, pose_3d_c