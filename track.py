import argparse
import torch
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from tracking import clip
from tracking.sort.detection import Detection
from tracking.sort.tracker import DeepSORTTracker
from tracking.utils import *

import sys
sys.path.insert(0, 'yolov5')
from yolov5.models.experimental import attempt_load



class Tracking:
    def __init__(self, 
        yolo_model_path, 
        img_size=640,
        filter_class=None,
        conf_thres=0.25,
        iou_thres=0.45,
        max_cosine_dist=0.4,    # the higher the value, the easier it is to assume it is the same person
        max_iou_dist=0.7,       # how much bboxes should overlap to determine the identity of the unassigned track
        nn_budget=None,         # indicates how many previous frames of features vectors should be retained for distance calc for ecah track
        max_age=60,             # specifies after how many frames unallocated tracks will be deleted
        n_init=3                # specifies after how many frames newly allocated tracks will be activated
    ) -> None:
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.filter_class = filter_class

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = attempt_load(yolo_model_path, map_location=self.device)
        self.model = self.model.to(self.device)
        self.names = self.model.names

        self.clip_model, self.clip_transform = clip.load('ViT-B/32', device=self.device, jit=False)
        self.tracker = DeepSORTTracker('cosine', max_cosine_dist, nn_budget, max_iou_dist, max_age, n_init)


    def preprocess(self, image):
        img = letterbox(image, new_shape=self.img_size)
        img = np.ascontiguousarray(img.transpose((2, 0, 1)))
        img = torch.from_numpy(img).to(self.device)
        img = img.float() / 255.0
        img = img[None]
        return img


    def extract_clip_features(self, boxes, img):
        image_patches = []
        for xyxy in boxes:
            x1, y1, x2, y2 = map(int, xyxy)
            img_patch = Image.fromarray(img[y1:y2, x1:x2])
            img_patch = self.clip_transform(img_patch)
            image_patches.append(img_patch)

        image_patches = torch.stack(image_patches).to(self.device)
        features = self.clip_model.encode_image(image_patches).cpu().numpy()
        return features


    def to_tlwh(self, boxes):
        boxes[:, 2] -= boxes[:, 0]
        boxes[:, 3] -= boxes[:, 1]
        return boxes


    def postprocess(self, pred, img1, img0):
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.filter_class)

        for det in pred:
            if len(det):
                boxes = scale_boxes(det[:, :4], img0.shape[:2], img1.shape[-2:]).cpu()
                features = self.extract_clip_features(boxes, img0)
                detections = [
                    Detection(bbox, class_id, feature) 
                for bbox, class_id, feature in zip(self.to_tlwh(boxes), det[:, 5], features)]

                self.tracker.predict()
                self.tracker.update(detections)

                for track in self.tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1: continue
                    label = f"{self.names[int(track.class_id)]} #{track.track_id}"
                    plot_one_box(track.to_tlbr(), img0, color=colors(int(track.class_id)), label=label)
            else:
                self.tracker.increment_ages()

        
    @torch.no_grad()
    def predict(self, image):
        img = self.preprocess(image)
        pred = self.model(img)[0]  
        self.postprocess(pred, img, image)
        return image


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='0')
    parser.add_argument('--yolo-model-path', type=str, default='checkpoints/yolov5s.pt')
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--filter-class', nargs='+', type=int, default=None)
    parser.add_argument('--conf-thres', type=float, default=0.4)
    parser.add_argument('--iou-thres', type=float, default=0.5)
    parser.add_argument('--max-cosine-dist', type=float, default=0.2)
    parser.add_argument('--max-iou-dist', type=int, default=0.7)
    parser.add_argument('--nn-budget', type=int, default=100)
    parser.add_argument('--max-age', type=int, default=70)
    parser.add_argument('--n-init', type=int, default=3)
    return parser.parse_args()


if __name__ == '__main__':
    args = argument_parser()
    tracking = Tracking(
        args.yolo_model_path, 
        args.img_size, 
        args.filter_class,
        args.conf_thres, 
        args.iou_thres, 
        args.max_cosine_dist,  
        args.max_iou_dist,
        args.nn_budget,
        args.max_age,
        args.n_init
    )

    if args.source.isnumeric():
        webcam = WebcamStream()
        fps = FPS()

        for frame in webcam:
            fps.start()
            output = tracking.predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            fps.stop()
            cv2.imshow('frame', cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
    
    else:
        reader = VideoReader(args.source)
        writer = VideoWriter(f"{args.source.rsplit('.', maxsplit=1)[0]}_out.mp4", reader.fps)

        for frame in tqdm(reader):
            output = tracking.predict(frame.numpy())
            writer.update(output)
        writer.write()
