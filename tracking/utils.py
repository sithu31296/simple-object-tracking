import cv2
import time
import random
from threading import Thread


class WebcamStream:
    def __init__(self, src=0) -> None:
        cap = cv2.VideoCapture(src)
        assert cap.isOpened(), f"Failed to open webcam {src}"
        _, self.frame = cap.read()
        Thread(target=self.update, args=([cap]), daemon=True).start()

    def update(self, cap):
        while cap.isOpened():
            cap.grab()
            _, self.frame = cap.retrieve()

    def read(self):
        return self.frame.copy()

    def stop(self):
        cv2.destroyAllWindows()
        raise StopIteration

class FPS:
    def __init__(self) -> None:
        self.accum_time = 0
        self.curr_fps = 0
        self.fps = "FPS: ??"

    def start(self):
        self.prev_time = time.time()

    def update(self):
        self.curr_time = time.time()
        self.accum_time += self.curr_time - self.prev_time
        self.prev_time = self.curr_time

    def get_fps(self):
        self.curr_fps += 1
        if self.accum_time > 1:
            self.accum_time -= 1
            self.fps = f"FPS: {self.curr_fps}"
            self.curr_fps = 0
        return self.fps


def plot_one_box(box, img, color=None, label=None):
    tl = round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    color = color or [random.randint(0, 255) for _ in range(3)]
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(img, p1, p2, color, tl, lineType=cv2.LINE_AA)

    if label:
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl/3, thickness=tf)[0]
        p2 = p1[0] + t_size[0], p1[1] - t_size[1] - 3
        cv2.rectangle(img, p1, p2, color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (p1[0], p1[1]-2), 0, tl/3, [255, 255, 255], thickness=tf, lineType=cv2.LINE_AA)