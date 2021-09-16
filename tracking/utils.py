import cv2
import time
import random
import torch
import os
import urllib.request
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torchvision import ops, io
from threading import Thread
from torch.backends import cudnn
cudnn.benchmark = True
cudnn.deterministic = False


def coco_class_index(class_name: str) -> int:
    coco_classes = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush'
    ]
    assert class_name.lower() in coco_classes, f"Invalid Class Name.\nAvailable COCO classes: {coco_classes}"
    return coco_classes.index(class_name.lower())


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()


class WebcamStream:
    def __init__(self, src=0) -> None:
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        assert self.cap.isOpened(), f"Failed to open webcam {src}"
        _, self.frame = self.cap.read()
        Thread(target=self.update, args=([]), daemon=True).start()

    def update(self):
        while self.cap.isOpened():
            _, self.frame = self.cap.read()

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1

        if cv2.waitKey(1) == ord('q'):
            self.stop()

        return self.frame.copy()

    def stop(self):
        cv2.destroyAllWindows()
        raise StopIteration

    def __len__(self):
        return 0


class SequenceStream:
    def __init__(self, folder): 
        self.frames = self.read_frames(folder)

        print(f"Processing '{folder}'...")
        print(f"Total Frames: {len(self.frames)}")
        print(f"Video Size  : {self.frames[0].shape[:-1]}")

    def read_frames(self, folder):
        files = sorted(list(Path(folder).glob('*.jpg')))
        frames = []
        for file in files:
            img = cv2.imread(str(file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)
        return frames

    def __iter__(self):
        self.count = 0
        return self

    def __len__(self):
        return len(self.frames)

    def __next__(self):
        if self.count == len(self.frames):
            raise StopIteration
        frame = self.frames[self.count]
        self.count += 1
        return frame
        

class VideoReader:
    def __init__(self, video: str):
        self.frames, _, info = io.read_video(video, pts_unit='sec')
        self.fps = info['video_fps']

        print(f"Processing '{video}'...")
        print(f"Total Frames: {len(self.frames)}")
        print(f"Video Size  : {list(self.frames.shape[1:-1])}")
        print(f"Video FPS   : {self.fps}")

    def __iter__(self):
        self.count = 0
        return self

    def __len__(self):
        return len(self.frames)

    def __next__(self):
        if self.count == len(self.frames):
            raise StopIteration
        frame = self.frames[self.count]
        self.count += 1
        return frame


class VideoWriter:
    def __init__(self, file_name, fps):
        self.fname = file_name
        self.fps = fps
        self.frames = []

    def update(self, frame):
        if isinstance(frame, np.ndarray):
            frame = torch.from_numpy(frame)
        self.frames.append(frame)

    def write(self):
        print(f"Saving video to '{self.fname}'...")
        io.write_video(self.fname, torch.stack(self.frames), self.fps)


class FPS:
    def __init__(self, avg=10) -> None:
        self.accum_time = 0
        self.counts = 0
        self.avg = avg

    def synchronize(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def start(self):
        self.synchronize()
        self.prev_time = time.time()

    def stop(self, debug=True):
        self.synchronize()
        self.accum_time += time.time() - self.prev_time
        self.counts += 1
        if self.counts == self.avg:
            self.fps = round(self.counts / self.accum_time)
            if debug: print(f"FPS: {self.fps}")
            self.counts = 0
            self.accum_time = 0


def plot_one_box(box, img, color=None, label=None):
    color = color or [random.randint(0, 255) for _ in range(3)]
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(img, p1, p2, color, 2, lineType=cv2.LINE_AA)

    if label:
        t_size = cv2.getTextSize(label, 0, fontScale=0.5, thickness=1)[0]
        p2 = p1[0] + t_size[0], p1[1] - t_size[1] - 3
        cv2.rectangle(img, p1, p2, color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (p1[0], p1[1]-2), 0, 0.5, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)


def letterbox(img, new_shape=(640, 640)):
    H, W = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / H, new_shape[1] / W)
    nH, nW = round(H * r), round(W * r)
    pH, pW = np.mod(new_shape[0] - nH, 32) / 2, np.mod(new_shape[1] - nW, 32) / 2

    if (H, W) != (nH, nW):
        img = cv2.resize(img, (nW, nH), interpolation=cv2.INTER_LINEAR)

    top, bottom = round(pH - 0.1), round(pH + 0.1)
    left, right = round(pW - 0.1), round(pW + 0.1)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img


def scale_boxes(boxes, orig_shape, new_shape):
    H, W = orig_shape
    nH, nW = new_shape
    gain = min(nH / H, nW / W)
    pad = (nH - H * gain) / 2, (nW - W * gain) / 2

    boxes[:, ::2] -= pad[1]
    boxes[:, 1::2] -= pad[0]
    boxes[:, :4] /= gain
    
    boxes[:, ::2].clamp_(0, orig_shape[1])
    boxes[:, 1::2].clamp_(0, orig_shape[0])
    return boxes.round()


def xywh2xyxy(x):
    boxes = x.clone()
    boxes[:, 0] = x[:, 0] - x[:, 2] / 2
    boxes[:, 1] = x[:, 1] - x[:, 3] / 2
    boxes[:, 2] = x[:, 0] + x[:, 2] / 2
    boxes[:, 3] = x[:, 1] + x[:, 3] / 2
    return boxes


def non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=None):
    candidates = pred[..., 4] > conf_thres 

    max_wh = 4096
    max_nms = 30000
    max_det = 300

    output = [torch.zeros((0, 6), device=pred.device)] * pred.shape[0]

    for xi, x in enumerate(pred):
        x = x[candidates[xi]]

        if not x.shape[0]: continue

        # compute conf
        x[:, 5:] *= x[:, 4:5]   # conf = obj_conf * cls_conf

        # box
        box = xywh2xyxy(x[:, :4])

        # detection matrix nx6
        conf, j = x[:, 5:].max(1, keepdim=True)
        x = torch.cat([box, conf, j.float()], dim=1)[conf.view(-1) > conf_thres]

        # filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # check shape
        n = x.shape[0]
        if not n: 
            continue
        elif n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # batched nms
        c = x[:, 5:6] * max_wh
        boxes, scores = x[:, :4] + c, x[:, 4]
        keep = ops.nms(boxes, scores, iou_thres)

        if keep.shape[0] > max_det:
            keep = keep[:max_det]

        output[xi] = x[keep]

    return output


def download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and os.path.isfile(download_target):
        return download_target

    print(f"Downloading model from {url}")
    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    return download_target


