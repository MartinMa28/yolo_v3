from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


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
        lines = [l for l in lines if not l.startswith('#')]
        lines = [l.strip() for l in lines]

        block = {}
        blocks = []

        for line in lines:
            if line.startswith('['):
                if len(block) > 0:
                    # If the block is not empty, adds it to the list.
                    blocks.append(block)
                    block = {}
                
                block['type'] = line[1: -1].strip()
            else:
                k, v = line.split('=')
                block[k.strip()] = v.strip()

        blocks.append(block)

        return blocks


def create_modules(blocks):
    net_info = blocks[0]
    module_list = nn.ModuleList()
    input_filters = 3
    output_filters = []

    for idx, b in enumerate(blocks[1:]):
        module = nn.Sequential()

        # check the type of the block
        if b['type'] == 'convolutional':
            activation = b['activation']
            try:
                batch_normalize = int(b['batch_normalize'])
                bias = False
            except:
                batch_normalize = 0
                bias = True
            
            filters = int(b['filters'])
            padding = int(b['pad'])
            kernel_size = int(b['size'])
            stride = int(b['stride'])

            if padding > 0:
                padding = (kernel_size - 1) / 2
            
            # Add the convolutional layer
            conv = nn.Conv2d(input_filters, filters, kernel_size=kernel_size,
            stride=stride, padding=padding, bias=bias)
            module.add_module('conv_{}'.format(idx), conv)

            if batch_normalize > 0:
                bn = nn.BatchNorm2d(filters)
                module.add_module('batch_norm_{}'.format(idx), bn)
            
            if activation == 'leaky':
                leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
                module.add_module('leaky_relu_{}'.format(idx), leaky_relu)
            
        elif b['type'] == 'upsample':
            stride = int(b['stride'])
            upsample = nn.Upsample(scale_factor=stride, mode='bilinear')
            module.add_module('upsample_{}'.format(idx), upsample)

        elif b['type'] == 'route':
            # if it is a route layer (skip connection)
            start_end = b['layers'].split(',')
            start = int(start_end[0])
            
            if len(start_end) > 1:
                end = int(start_end[1])
            else:
                end = 0

            if start > 0:
                start = start - idx

            if end > 0:
                end = end - idx

            route = EmptyLayer()
            module.add_module('route_{}'.format(idx), route)

            if end < 0:
                filters = output_filters[idx + start] + output_filters[idx + end]
            else:
                # no end
                filters = output_filters[idx + start]
        
        elif b['type'] == 'shortcut':
            shortcut = EmptyLayer()
            module.add_module('shortcut_{}'.format(idx), shortcut)

        elif b['type'] == 'yolo':
            mask = b['mask'].split(',')
            mask = [int(x) for x in mask]

            anchors = b['anchors'].split(',')
            anchors = [int(x) for x in anchors]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(len(anchors))[::2]]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module('detection_{}'.format(idx), detection)

        module_list.append(module)
        input_filters = filters
        output_filters.append(filters)

    return (net_info, module_list)
            


if __name__ == "__main__":
    blocks = parse_cfg('cfg/yolov3.cfg')
    for b in blocks:
        print(b)
