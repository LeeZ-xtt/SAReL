"""Feature extractor and classifier used in SAReL."""

import math

import torch
import torch.nn as nn


def init_layer(layer):
  if isinstance(layer, nn.Conv1d):
    n = layer.kernel_size[0] * layer.out_channels
    layer.weight.data.normal_(0, math.sqrt(2.0 / float(n)))
  elif isinstance(layer, nn.BatchNorm1d):
    layer.weight.data.fill_(1)
    layer.bias.data.fill_(0)


class Flatten(nn.Module):
  def forward(self, x):
    return x.view(x.size(0), -1)


class DeepINClassifierFeat(nn.Module):
  """CNN4 backbone split after block 2 for statistic augmentation."""

  input_length = 1024

  def __init__(
    self,
    num_classes,
    dropout=0.0,
    head_type="linear",
    head_proj_dim=0,
    **kwargs,
  ):
    super().__init__()
    block12 = [
      nn.Conv1d(1, 32, kernel_size=16, stride=1),
      nn.BatchNorm1d(32),
      nn.ReLU(inplace=True),
      nn.MaxPool1d(2),
      nn.Conv1d(32, 64, kernel_size=5, stride=1),
      nn.BatchNorm1d(64),
      nn.ReLU(inplace=True),
      nn.MaxPool1d(2),
    ]
    block34 = [
      nn.Conv1d(64, 96, kernel_size=5, stride=1),
      nn.BatchNorm1d(96),
      nn.ReLU(inplace=True),
      nn.MaxPool1d(2),
      nn.Conv1d(96, 128, kernel_size=3, stride=1),
      nn.BatchNorm1d(128),
      nn.ReLU(inplace=True),
      nn.MaxPool1d(2),
      Flatten(),
    ]
    for layer in block12 + block34:
      if isinstance(layer, (nn.Conv1d, nn.BatchNorm1d)):
        init_layer(layer)
    self.block12 = nn.Sequential(*block12)
    self.block34 = nn.Sequential(*block34)
    self.dropout = nn.Dropout(p=float(dropout))
    self.final_feat_dim = self._infer_final_feat_dim()
    self.head_type = str(head_type).lower()
    if self.head_type == "linear128":
      proj_dim = 128
    elif int(head_proj_dim) > 0:
      proj_dim = int(head_proj_dim)
    else:
      proj_dim = 0
    self.head_proj_dim = proj_dim

    if self.head_proj_dim > 0:
      self.projector = nn.Linear(self.final_feat_dim, self.head_proj_dim, bias=False)
      clf_in_dim = self.head_proj_dim
    else:
      self.projector = None
      clf_in_dim = self.final_feat_dim

    self.classifier = nn.Linear(clf_in_dim, int(num_classes))

  def _infer_final_feat_dim(self):
    was_training = (self.block12.training, self.block34.training)
    self.block12.eval()
    self.block34.eval()
    with torch.no_grad():
      dummy = torch.zeros(1, 1, self.input_length)
      features = self.block34(self.block12(dummy))
    self.block12.train(was_training[0])
    self.block34.train(was_training[1])
    return int(features.shape[1])

  def forward_block12(self, x):
    return self.block12(x)

  def forward_block34_feat(self, feat12):
    return self.block34(feat12)

  def forward_classifier(self, feat34, y=None):
    feat = self.dropout(feat34)
    if self.projector is not None:
      feat = self.projector(feat)
    return self.classifier(feat)

  def forward_block34(self, feat12, y=None):
    return self.forward_classifier(self.forward_block34_feat(feat12), y=y)

  def forward(self, x, y=None):
    feat12 = self.forward_block12(x)
    return self.forward_block34(feat12, y=y)
