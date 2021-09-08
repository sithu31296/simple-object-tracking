# <div align="center">Simple Object Tracking</div>

<div align="center">
<p>Multi-Object Tracking with YOLOv5, CLIP and DeepSORT</p>
<p>
<img src="gifs/test_out.gif" width="270"/> <img src="gifs/newyork_out.gif" width="270"/> <img src="gifs/cars_out.gif" width="270"/> 
</p>
</div>

## Introduction

This is a simple two-stage mulit-object tracking with zero-shot or self-supervised feature extractors. This implementation is based from Roboflow [zero-shot-object-tracking](https://github.com/roboflow-ai/zero-shot-object-tracking); which incorporates CLIP as a feature extractor in DeepSORT. 

CLIP is a zero-shot classification model; which is pretrained under vision-langauge supervision with a lot of data. CLIP's zero-shot performance is on par with supervised ResNet models.

The benefit of this approach is that it can track a lof of classes out-of-the-box without needing to re-train the feature extractor (re-identification model) for a specific class in DeepSORT. Its performance may not be as good as traditional re-identification model; which is trained on tracking/re-identification datasets.

## Supported Models and Trackers

Feature Extractors (Replacement of Supervised Re-ID Models)

- [x] [CLIP](https://arxiv.org/abs/2103.00020) (Zero-shot)
- [x] [DINO](https://arxiv.org/abs/2104.14294v2) (SSL)

Trackers

- [x] [DeepSORT](https://arxiv.org/abs/1703.07402)
- [ ] [Tracktor++](https://arxiv.org/abs/1903.05625)
- [ ] [UniTrack](https://arxiv.org/abs/2107.02156)

## Requirements

* torch >= 1.8.1
* torchvision >= 0.9.1

Other requirements can be installed with `pip install -r requirements.txt`.

Clone the repository recursively:

```bash
$ git clone --recursive https://github.com/sithu31296/simple-object-tracking.git
```

Then download a YOLO model's weight from [YOLOv5](https://github.com/ultralytics/yolov5) and place it in `checkpoints`.

## Tracking

Track all classes:

```bash
## webcam
$ python track.py --source 0 --yolo-model checkpoints/yolov5s.pt --reid-model CLIP-RN50

## video
$ python track.py --source VIDEO_PATH --yolo-model checkpoints/yolov5s.pt --reid-model CLIP-RN50
```

Track only specified classes:

```bash
## track only person class
$ python track.py --source 0 --yolo-model checkpoints/yolov5s.pt --reid-model CLIP-RN50 --filter-class 0

## track person and car classes
$ python track.py --source 0 --yolo-model checkpoints/yolov5s.pt --reid-model CLIP-RN50 --filter-class 0 2
```

Available ReID models (Feature Extractors):
* **CLIP**: `CLIP-RN50`, `CLIP-ViT-B/32`
* **DINO**: `DINO-XciT-S12/16`, `DINO-XciT-S12/8`, `DINO-XciT-M24/16`, `DINO-ViT-S/16`, `DINO-ViT-S/8`, `DINO-ViT-B/16`

Check [here](tracking/utils.py#L14) to get COCO class index for your class.

## Evaluate on MOT16

* Download MOT16 dataset from [here](https://motchallenge.net/data/MOT16.zip) and unzip it.
* Download mot-challenge ground-truth [data](https://omnomnom.vision.rwth-aachen.de/data/TrackEval/data.zip) for evaluating with TrackEval. Then, unzip it under the project directory.
* Save the tracking results of MOT16 with the following command:

```bash
$ python eval_mot.py --root MOT16_ROOT_DIR --yolo-model checkpoints/yolov5m.pt --reid-model CLIP-RN50
```

* Evaluate with TrackEval:

```bash
$ python TrackEval/scripts/run_mot_challenge.py
    --BENCHMARK MOT16
    --GT_FOLDER PROJECT_ROOT/data/gt/mot_challenge/
    --TRACKERS_FOLDER PROJECT_ROOT/data/trackers/mot_challenge/
    --TRACKERS_TO_EVAL mot_det
    --SPLIT_TO_EVAL train
    --USE_PARALLEL True
    --NUM_PARALLEL_CORES 4
    --PRINT_ONLY_COMBINED True
```

> Notes: `FOLDER` parameters in `run_mot_challenge.py` must be an absolute path.

For tracking persons, instead of using a COCO-pretrained model, using a model trained on multi-person dataset will get better accuracy. You can download a YOLOv5m model trained on [CrowdHuman](https://www.crowdhuman.org/) dataset from [here](https://drive.google.com/file/d/1gglIwqxaH2iTvy6lZlXuAcMpd_U0GCUb/view?usp=sharing). The weights are from [deepakcrk/yolov5-crowdhuman](https://github.com/deepakcrk/yolov5-crowdhuman). It has 2 classes: 'person' and 'head'. So, you can use this model for both person and head tracking.

## Results

**MOT16 Evaluation Results**

Detector | Feature Extractor | MOTA↑ | HOTA↑ | IDF1↑ | IDsw↓ | MT↑ | ML↓ | FP↓ | FN↓ | FPS<br><sup>(GTX1660ti)
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---
YOLOv5m<sup><br>(COCO) | CLIP<sup><br>(RN50) | 35.42 | 35.37 | 39.42 | **486** | 115 | 192 | **6880** | 63931 | 7
YOLOv5m<sup><br>(CrowdHuman) | CLIP<sup><br>(RN50) | 53.25 | 43.25 | 52.12 | 912 | 196 | **89** | 14076 | 36625 | 6
YOLOv5m<sup><br>(CrowdHuman) | CLIP<sup><br>(ViT-B/32) | 53.35 | 43.03 | 51.25 | 896 | **199** | 91 | 14035 | **36575** | 4
||
YOLOv5m<sup><br>(CrowdHuman) | DINO<sup><br>(XciT-S12/16) | 54.41 | 47.44 | 59.01 | 511 | 184 | 101 | 12265 | 37555 |8
YOLOv5m<sup><br>(CrowdHuman) | DINO<sup><br>(XciT-S12/8) | 54.44 | 47.63 | 59.24 | 517 | 185 | 98 | 12140 | 37639 | 4
YOLOv5m<sup><br>(CrowdHuman) | DINO<sup><br>(XciT-M24/16) | 54.56 | **47.71** | **59.77** | 504 | 187 | 96 | 12364 | 37306 | 5
YOLOv5m<sup><br>(CrowdHuman) | DINO<sup><br>(ViT-S/16) | 54.56 | 47.61 | 58.94 | 519 | 189 | 97 | 12346 | 37308 | 8
YOLOv5m<sup><br>(CrowdHuman) | DINO<sup><br>(ViT-S/8) | 54.53 | 47.70 | 59.20 | 542 | 180 | 102 | 11912 | 37744 | 4
YOLOv5m<sup><br>(CrowdHuman) | DINO<sup><br>(ViT-B/16) | **54.58** | 47.55 | 58.89 | 507 | 184 | 97 | 12017 | 37621 | 5

**FPS Results**

Detector | Feature Extractor | GPU | Precision | Image Size | Detection<br>/Frame | FPS
--- | --- | --- | --- | --- | --- | ---
YOLOv5s | CLIP (RN50) | GTX-1660ti | FP32 | 480x640 | 1 | 40
YOLOv5m | CLIP (RN50) | GTX-1660ti | FP32 | 480x640 | 1 | 32
YOLOv5s | CLIP (ViT-B/32) | GTX-1660ti | FP32 | 480x640 | 1 | 30
YOLOv5m | CLIP (ViT-B/32) | GTX-1660ti | FP32 | 480x640 | 1 | 23


## References

* https://github.com/roboflow-ai/zero-shot-object-tracking
* https://github.com/ultralytics/yolov5
* https://github.com/openai/CLIP
* https://github.com/JonathonLuiten/TrackEval
* https://github.com/deepakcrk/yolov5-crowdhuman

## Citations

```
@inproceedings{caron2021emerging,
  title={Emerging Properties in Self-Supervised Vision Transformers},
  author={Caron, Mathilde and Touvron, Hugo and Misra, Ishan and J\'egou, Herv\'e  and Mairal, Julien and Bojanowski, Piotr and Joulin, Armand},
  booktitle={Proceedings of the International Conference on Computer Vision (ICCV)},
  year={2021}
}

@misc{yolov5deepsort2020,
  title={Real-time multi-object tracker using YOLOv5 and deep sort},
  author={Mikel Broström},
  howpublished = {\url{https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch}},
  year={2020}
}

@article{el2021xcit,
  title={XCiT: Cross-Covariance Image Transformers},
  author={El-Nouby, Alaaeldin and Touvron, Hugo and Caron, Mathilde and Bojanowski, Piotr and Douze, Matthijs and Joulin, Armand and Laptev, Ivan and Neverova, Natalia and Synnaeve, Gabriel and Verbeek, Jakob and others},
  journal={arXiv preprint arXiv:2106.09681},
  year={2021}
}

@inproceedings{Wojke2017simple,
  title={Simple Online and Realtime Tracking with a Deep Association Metric},
  author={Wojke, Nicolai and Bewley, Alex and Paulus, Dietrich},
  booktitle={2017 IEEE International Conference on Image Processing (ICIP)},
  year={2017},
  pages={3645--3649},
  organization={IEEE},
  doi={10.1109/ICIP.2017.8296962}
}

@inproceedings{Wojke2018deep,
  title={Deep Cosine Metric Learning for Person Re-identification},
  author={Wojke, Nicolai and Bewley, Alex},
  booktitle={2018 IEEE Winter Conference on Applications of Computer Vision (WACV)},
  year={2018},
  pages={748--756},
  organization={IEEE},
  doi={10.1109/WACV.2018.00087}
}
```