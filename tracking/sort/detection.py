import numpy as np


class Detection:
    """Bounding box detection in a single image
    Parameters
    ----------
    tlwh        : (ndarray) bbox in format `(top left x, top left y, width, height)`.
    confidence  : (float) Detector confidence score.
    class_id    : (ndarray) Detector class.
    feature     : (ndarray) A feature vector that describes the object contained in this image.
    """
    def __init__(self, tlwh, class_id, feature):
        self.tlwh = np.asarray(tlwh, dtype=np.float32)
        self.feature = np.asarray(feature, dtype=np.float32)
        self.class_id = class_id

    def to_tlbr(self):
        """Convert bbox from (top, left, width, height) to (top, left, bottom, right)
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        """Convert bbox from (top, left, width, height) to (center x, center y, aspect ratio, height) where the aspect ratio is `width / height`
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret