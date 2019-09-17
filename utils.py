from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def predict_transform(prediction, input_dim, anchors, num_classes):
    
    batch_size = prediction.size(0)
    stride = input_dim // prediction.size(2)
    grid_size = input_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    prediction = prediction.view(batch_size, num_anchors * bbox_attrs, grid_size * grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(batch_size, grid_size * grid_size * num_anchors, bbox_attrs)
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    # sigmoid the center x
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    # sigmoid the center y
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    # sigmoid the object confidence
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])

    # Add the center offsets
    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)

    if torch.cuda.is_available():
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)
    prediction[:, :, :2] += x_y_offset

    # log space transform height and width
    anchors = torch.FloatTensor(anchors)

    if torch.cuda.is_available():
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    prediction[:, :, 2: 4] = torch.exp(prediction[:, :, 2: 4]) * anchors
