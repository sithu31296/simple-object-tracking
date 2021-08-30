# Simple Object Tracking

**Object Tracking with YOLOv5, CLIP and DeepSORT**

## Introduction

This is the simplest implementation of Roboflow [zero-shot-object-tracking](https://github.com/roboflow-ai/zero-shot-object-tracking). 

It includes object detection model and clip+sort (replacing deepsort in normal object tracking). The benefit of this is that it can track a lot classes out-of-the-box without needing to re-train the deep model of the deepsort algorithm. [CLIP](https://openai.com/blog/clip/) is a zero-shot classification model; pretrained under vision-langauge supervision.


## Requirements

* torch >= 1.8.1
* torchvision >= 0.9.1

Other requirements can be installed with `pip install -r requirements.txt`.

Clone the repository recursively:

```bash
$ git clone --recurse-submodules https://github.com/sithu31296/simple-object-tracking.git
```

Then download a YOLO model's weight from [YOLOv5](https://github.com/ultralytics/yolov5) and place it in `checkpoints`.

## Tracking

The following command will open a webcam and start object tracking:

```bash
## track all COCO classes
$ python track.py --yolo-model-path 'checkpoints/yolov5s.pt'

## track only person class
$ python track.py --yolo-model-path 'checkpoints/yolov5s.pt' --filter-class 0
```

## References

* https://github.com/roboflow-ai/zero-shot-object-tracking
* https://github.com/ultralytics/yolov5
* https://openai.com/blog/clip/
