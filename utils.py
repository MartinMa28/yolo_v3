from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)

    # tensor_res = tensor.new(unique_tensor.shape)
    # tensor_res.copy_(unique_tensor)

    # return tensor_res

    return unique_tensor

def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes.
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]

    # get the corrdinates of the intersection rectangle:
    # 1. get the max of top-left x and y coordinates
    # 2. get the min of bottom-right x and y coordinates
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)

    # intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0)\
                    * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    # union area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    union_area = b1_area + b2_area - inter_area

    iou = inter_area / union_area

    return iou

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

    # put the class score through sigmoid
    prediction[:, :, 5: 5 + num_classes] = torch.sigmoid((prediction[:, :, 5: 5 + num_classes]))

    prediction[:, :, :4] *= stride

    return prediction


def write_results(prediction, confidence, num_classes, nms_conf = 0.4):
    # The fifth index of prediction output is the object confidence.
    # If it's below the threshold, set the values of its attributes to zero.
    
    conf_mask = (prediction[:, :, 4] > confidence).float().unsqueeze(2)
    prediction = prediction * conf_mask

    # transform (center x, center y, height, width)
    # to (top-left corner x, top-left corner y, bottom-right corner x, bottom-right corner y)
    box_corner = prediction.new(prediction.shape)
    box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2) 
    box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    prediction[:,:,:4] = box_corner[:,:,:4]
    
    batch_size = prediction.size(0)

    write = False

    for idx in range(batch_size):
        # get the image tensor
        image_pred = prediction[idx]

        max_confi, max_confi_score = torch.max(image_pred[:, 5: 5 + num_classes], 1)
        max_confi = max_confi.float().unsqueeze(1)
        max_confi_score = max_confi_score.float().unsqueeze(1)
        # seq has 7 columns: 
        # top left x & y, bottom right x & y, obj_confi, max_cls_confi, max_cls_confi_idx
        seq = (image_pred[:, :5], max_confi, max_confi_score)
        image_pred = torch.cat(seq, 1)

        non_zero_idx = torch.nonzero(image_pred[:, 4])
        image_pred_ = image_pred[non_zero_idx.squeeze()].view(-1, 7)
        

        if image_pred_.shape[0] == 0:
            # If no detection in this batch, skip the rest of loop body.
            continue

        # Get the unique classes detected in the image.
        # The last column is the max_cls_confi_idx, indicating the detected class.
        img_classes = unique(image_pred_[:, -1])

        for cl in img_classes:
            # Non-maximum suppression starts from here.

            # Extract detections of cl-class.
            cls_mask = image_pred_ * (image_pred_[:, -1] == cl).float().unsqueeze(1)
            class_mask_idx = torch.nonzero(cls_mask[:, -2]).squeeze()
            image_pred_class = image_pred_[class_mask_idx].view(-1, 7)

            # Sort the detections such that the entry with the maximum objectness.
            confi_sort_idx = torch.sort(image_pred_class[:, 4], descending=True)
            image_pred_class = image_pred_class[confi_sort_idx]
            # number of remaining detections
            num_detections = image_pred_class.size(0)

            for i in range(num_detections):
                # Get the IoUs of all boxes that come after the 
                # one we are looking for in the loop.
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
                except ValueError:
                    break
            
                except IndexError:
                    break
            
                #Zero out all the detections that have IoU > treshhold
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i+1:] *= iou_mask       
            
                #Remove the non-zero entries
                non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1,7)

            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)      #Repeat the batch_id for as many detections of the class cls in the image
            seq = batch_ind, image_pred_class
            
            if not write:
                output = torch.cat(seq,1)
                write = True
            else:
                out = torch.cat(seq,1)
                output = torch.cat((output, out))

    return output