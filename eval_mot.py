import argparse
import torch
import shutil
from pathlib import Path
from tqdm import tqdm
from tracking.utils import *

from track import Tracking


class EvalTracking(Tracking):
    def __init__(self, yolo_model, reid_model, img_size, filter_class, conf_thres, iou_thres, max_cosine_dist, max_iou_dist, nn_budget, max_age, n_init) -> None:
        super().__init__(yolo_model, reid_model, img_size=img_size, filter_class=filter_class, conf_thres=conf_thres, iou_thres=iou_thres, max_cosine_dist=max_cosine_dist, max_iou_dist=max_iou_dist, nn_budget=nn_budget, max_age=max_age, n_init=n_init)

    def postprocess(self, pred, img1, img0, txt_path, frame_idx):
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.filter_class)

        for det in pred:
            if len(det):
                boxes = scale_boxes(det[:, :4], img0.shape[:2], img1.shape[-2:]).cpu()
                features = self.extract_features(boxes, img0)

                self.tracker.predict()
                self.tracker.update(boxes, det[:, 5], features)

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
    parser.add_argument('--yolo-model', type=str, default='checkpoints/crowdhuman_yolov5m.pt')
    parser.add_argument('--reid-model', type=str, default='CLIP-RN50')
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
    tracking = EvalTracking(
        args.yolo_model,
        args.reid_model, 
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
        shutil.rmtree(save_path)
    save_path.mkdir(parents=True)

    root = Path(args.root) / 'train'
    folders = root.iterdir()

    total_fps = []

    for folder in folders:
        tracking.tracker.reset()
        reader = SequenceStream(folder / 'img1')
        txt_path = save_path / f"{folder.stem}.txt"
        fps = FPS(len(reader.frames))

        for i, frame in tqdm(enumerate(reader), total=len(reader)):
            fps.start()
            tracking.predict(frame, txt_path, i)
            fps.stop(False)

        print(f"FPS: {fps.fps}")
        total_fps.append(fps.fps)
        del reader
    
    print(f"Average FPS for MOT16: {round(sum(total_fps) / len(total_fps))}")