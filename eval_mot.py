import argparse
import torch
from pathlib import Path
from tqdm import tqdm
from tracking.sort.detection import Detection
from tracking.utils import *

from .track import Tracking


class EvalTracking(Tracking):
    def __init__(self, yolo_model_path, img_size, filter_class, conf_thres, iou_thres, max_cosine_dist, max_iou_dist, nn_budget, max_age, n_init) -> None:
        super().__init__(yolo_model_path, img_size=img_size, filter_class=filter_class, conf_thres=conf_thres, iou_thres=iou_thres, max_cosine_dist=max_cosine_dist, max_iou_dist=max_iou_dist, nn_budget=nn_budget, max_age=max_age, n_init=n_init)

    def postprocess(self, pred, img1, img0, txt_path, frame_idx):
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
                    
                    x1, y1, x2, y2 = track.to_tlbr()
                    w, h = x2 - x1, y2 - y1

                    with open(txt_path, 'a') as f:
                        f.write(f"{frame_idx+1},{track.track_id},{x1:.4f},{y1:.4f},{w:.4f},{h:.4f},-1,-1,-1,-1\n")
            else:
                self.tracker.increment_ages()

    @torch.no_grad()
    def predict(self, image, txt_path, frame_idx):
        img = self.preprocess(image)
        pred = self.model(img)[0]
        self.postprocess(pred, img, image, txt_path, frame_idx)


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/home/sithu/datasets/MOT16')
    parser.add_argument('--yolo-model-path', type=str, default='checkpoints/yolov5m.pt')
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--filter-class', nargs='+', type=int, default=0)
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

    save_path = Path('data') / 'trackers' / 'mot_challenge' / 'MOT16-train' / 'mot_det' / 'data'
    if save_path.exists():
        save_path.rmdir()
    save_path.mkdir(parents=True)

    root = Path(args.root) / 'train'
    folders = root.iterdir()

    for folder in folders:
        reader = SequenceStream(folder / 'img1')
        txt_path = save_path / f"{folder.stem}.txt"

        for i, frame in tqdm(enumerate(reader), total=len(reader)):
            tracking.predict(frame, txt_path, i)

        del reader