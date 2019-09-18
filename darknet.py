from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

from utils import predict_transform


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
                padding = (kernel_size - 1) // 2
            
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
            upsample = nn.Upsample(scale_factor=stride, mode='bilinear', align_corners=True)
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
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module('detection_{}'.format(idx), detection)

        module_list.append(module)
        input_filters = filters
        output_filters.append(filters)

    return (net_info, module_list)
            

class Darknet(nn.Module):
    def __init__(self, cfg_path):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfg_path)
        self.net_info, self.module_list = create_modules(self.blocks)

    
    def forward(self, x):
        modules = self.blocks[1:]
        outputs = {}

        write = 0
        for i, module in enumerate(modules):
            module_type = module['type']

            if module_type == 'convolutional' or \
                module_type == 'upsample':
                x = self.module_list[i](x)

            elif module_type == 'route':
                layers = module['layers'].split(',')
                layers = [int(a) for a in layers]

                if layers[0] > 0:
                    layers[0] = layers[0] - i

                if len(layers) == 1:
                    x = outputs[i + layers[0]]
                else:
                    if layers[1] > 0:
                        layers[1] = layers[1] - i

                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]

                    x = torch.cat((map1, map2), 1)
            
            elif module_type == 'shortcut':
                from_ = int(module['from'])
                x = outputs[i - 1] + outputs[i + from_]

            elif module_type == 'yolo':
                anchors = self.module_list[i][0].anchors
                # get the input dimension
                input_dim = int(self.net_info['height'])
                # get the number of classes
                num_classes = int(module['classes'])

                # transform
                x = predict_transform(x, input_dim, anchors, num_classes)
                if not write:
                    detections = x
                    write = 1
                else:
                    detections = torch.cat((detections, x), 1)

            outputs[i] = x
        
        return detections
    
    def load_weights(self, weights_path):
        with open(weights_path, 'rb') as fp:
            # The first 5 values are header info:
            # 1. major version number
            # 2. minor version number
            # 3. subversion number
            # 4 / 5. images seen by the network during the training
            header = np.fromfile(fp, dtype=np.int32, count=5)
            self.header = torch.from_numpy(header)
            self.seen = self.header[3]

            weights = np.fromfile(fp, dtype=np.float32)
            
            ptr = 0
            for i, module in enumerate(self.module_list):
                module_type = self.blocks[i + 1]['type']
                if module_type == 'convolutional':
                    try:
                        batch_norm = int(self.blocks[i + 1]['batch_normalize'])
                    except:
                        batch_norm = 0

                    conv = module[0]

                    if batch_norm > 0:
                        bn = module[1]

                        # Gets the number of weights of batch norm layer.
                        num_bn_biases = bn.bias.numel()

                        # Loads the weights.
                        bn_biases = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                        ptr += num_bn_biases

                        bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                        ptr += num_bn_biases

                        bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                        ptr += num_bn_biases

                        bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                        ptr += num_bn_biases

                        # Casts the loaded weights into dimensions of the model weights.
                        bn_biases = bn_biases.view_as(bn.bias.data)
                        bn_weights = bn_weights.view_as(bn.weight.data)
                        bn_running_mean = bn_running_mean.view_as(bn.running_mean.data)
                        bn_running_var = bn_running_var.view_as(bn.running_var.data)

                        # Copies the data to model
                        bn.bias.data.copy_(bn_biases)
                        bn.weight.data.copy_(bn_weights)
                        bn.running_mean.data.copy_(bn_running_mean)
                        bn.running_var.data.copy_(bn_running_var)
                    
                    else:
                        # number of biases
                        num_biases = conv.bias.numel()

                        # Loads the weights.
                        conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                        ptr += num_biases

                        # Reshapes the loaded weights according to the dimension of
                        # the model weights.
                        conv_biases = conv_biases.view_as(conv.bias.data)

                        # Copies the data, finally.
                        conv.bias.data.copy_(conv_biases)

                    # After loading bias, loads convolutional weights.
                    num_weights = conv.weight.numel()
                    conv_weights = torch.from_numpy(weights[ptr: ptr + num_weights])
                    ptr += num_weights

                    conv_weights = conv_weights.view_as(conv.weight.data)
                    conv.weight.data.copy_(conv_weights)



def get_test_input():
    img = cv2.imread('dog-cycle-car.png')
    img = cv2.resize(img, (416, 416))
    # BGR -> RGB, HxWxC -> CxHxW
    img = img[:, :,::-1].transpose((2, 0, 1))
    img = img[np.newaxis, :, :, :] / 255
    img = torch.from_numpy(img).float()

    return img


if __name__ == "__main__":
    model = Darknet('cfg/yolov3.cfg')
    model.load_weights('yolov3.weights')
    model.cuda()
    inp = get_test_input()
    if torch.cuda.is_available():
        inp = inp.cuda()
    pred = model(inp)
    print(pred.shape)
