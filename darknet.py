from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def parse_cfg(cfg_path):
    """
    Args:
        cfg_path: the cfg file path

    Returns:
        Returns a list of blocks. Each block is corresponding to a building block
        of the yolo v3. Block is represented as a dictionary in the list.
    """
    with open(cfg_path) as cfg_file:
        lines = cfg_file.read().split('\n')
        # remove empty lines, comments and whitespaces
        lines = [l for l in lines if len(l) > 0]