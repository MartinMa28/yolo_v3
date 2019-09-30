# yolo_v3
The implementation of yolo v3 for congested vehicle detection using PyTorch

## YOLOv3 model
The model is implemented by parsing darknet config file. The model is tested. It's in darknet.py.

## object detection related functions
The functions related to object detection techniques reside in utils.py, including  transforming the the model outputs with the anchor boxes, calculating the bounding  box from the model outputs, calculating IoU using the bounding box, and finally conduct the non-max suppression over a the detections of a certain class. **This part has not been tested. Some codes look strange, needed to be cleaned up and refactorized.**

## To be done
The input and output pipeline.