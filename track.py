import argparse
import torch
import cv2
import numpy as np
from PIL import Image

from tracking import clip
from tracking.sort.detection import Detection
from tracking.sort.nn_matching import NearestNeighborDistanceMetric
from tracking.sort.tracker import Tracker
from tracking.utils import WebcamStream, FPS, plot_one_box

import sys
sys.path.insert(0, 'yolov5')
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.datasets import letterbox



class Tracking:
    def __init__(self, 
        yolo_model_path, 
        img_size=(480, 640),
        conf_thres=0.25,
        iou_thres=0.45,
        max_cosine_dist=0.4,
        nn_budget=None,
        filter_class=None
    ) -> None:
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.filter_class = filter_class

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = attempt_load(yolo_model_path, map_location=self.device)
        self.model = self.model.to(self.device)
        self.names = self.model.names

        self.clip_model, self.clip_transform = clip.load('ViT-B/32', device=self.device, jit=True)
        self.tracker = Tracker(NearestNeighborDistanceMetric('cosine', max_cosine_dist, nn_budget))


    def preprocess(self, image):
        img = letterbox(image, new_shape=self.img_size)[0]
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float() / 255.0
        img = img[None]
        return img


    def extract_clip_features(self, det, img):
        image_patches = []
        for *xyxy, _, _ in reversed(det):
            x1, y1, x2, y2 = list(map(lambda x: int(x.item()), xyxy))
            img_patch = Image.fromarray(img[y1:y2, x1:x2])
            img_patch = self.clip_transform(img_patch)
            image_patches.append(img_patch)

        image_patches = torch.stack(image_patches).to(self.device)
        features = self.clip_model.encode_image(image_patches).cpu().numpy()
        return features


    def to_tlwh(self, boxes):
        boxes2 = boxes.clone().cpu()
        boxes2[:, 2] -= boxes2[:, 0]
        boxes2[:, 3] -= boxes2[:, 1]
        return boxes2


    def postprocess(self, pred, img1, img0):
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.filter_class)

        for det in pred:
            if len(det):
                det[:, :4] = scale_coords(img1.shape[-2:], det[:, :4], img0.shape[:2]).round()

                boxes = self.to_tlwh(det[:, :4])
                features = self.extract_clip_features(det, img0)

                detections = [
                    Detection(bbox, conf, class_num, feature) 
                for bbox, conf, class_num, feature in zip(boxes, det[:, 4], det[:, 5], features)]

                self.tracker.predict()
                self.tracker.update(detections)

                for track in self.tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue

                    label = f"{self.names[int(track.class_num)]} #{track.track_id}"
                    plot_one_box(track.to_tlbr(), img0, color=[0, 255, 0], label=label)

        
    @torch.no_grad()
    def predict(self, image):
        img = self.preprocess(image)
        pred = self.model(img)[0]  
        self.postprocess(pred, img, image)
        return image


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-model-path', type=str, default='checkpoints/yolov5s.pt')
    parser.add_argument('--img-size', type=tuple, default=(480, 640))
    parser.add_argument('--conf-thres', type=float, default=0.25)
    parser.add_argument('--iou-thres', type=float, default=0.45)
    parser.add_argument('--max-cosine-dist', type=float, default=0.4)
    parser.add_argument('--nn-budget', type=int, default=None)
    parser.add_argument('--filter-class', type=int, default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = argument_parser()
    tracking = Tracking(
        args.yolo_model_path, 
        args.img_size, 
        args.conf_thres, 
        args.iou_thres, 
        args.max_cosine_dist, 
        args.nn_budget, 
        args.filter_class
    )
    webcam = WebcamStream()
    fps = FPS()

    while True:
        fps.start()
        frame = webcam.read()
        vis = tracking.predict(frame)
        fps.update()
        print(fps.get_fps())

        cv2.imshow('frame', vis)

        if cv2.waitKey(1) == ord('q'):
            webcam.stop()
