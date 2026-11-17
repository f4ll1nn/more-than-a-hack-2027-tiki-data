"""Sparse native-frame decoding and frozen single-class CUDA ball detection."""

import hashlib
from importlib import metadata

import numpy as np


WEIGHTS_SHA256 = "678fbad05134f19c5094cb8d273812ec9c6691228180d46832551ecf99ed2912"
RUNTIME = {
    "numpy": "2.5.4",
    "opencv-python-headless": "4.11.0.86",
    "torch": "2.7.1+cu128",
    "torchvision": "0.22.1+cu128",
    "ultralytics": "8.4.208",
}


def read_frames(path, ticks, frame_count):
    import cv2

    options = [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_NONE,
               cv2.CAP_PROP_N_THREADS, 2]
    capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, options)
    try:
        if (not capture.isOpened()
                or capture.get(cv2.CAP_PROP_HW_ACCELERATION) != cv2.VIDEO_ACCELERATION_NONE
                or abs(capture.get(cv2.CAP_PROP_FPS) - 25) > 1e-6
                or round(capture.get(cv2.CAP_PROP_FRAME_COUNT)) != frame_count):
            raise ValueError("expected the 25 Hz tactical video in the software decoder")
        cursor = 0
        for tick in ticks:
            frame = tick * 5
            if frame - cursor > 25:
                if not capture.set(cv2.CAP_PROP_POS_FRAMES, frame):
                    raise ValueError(f"could not seek to native frame {frame}")
                cursor = frame
            while cursor < frame:
                if not capture.grab():
                    raise ValueError(f"could not advance to native frame {frame}")
                cursor += 1
            ok, image = capture.read()
            if (not ok or image.shape != (1080, 1920, 3) or image.dtype != np.uint8
                    or round(capture.get(cv2.CAP_PROP_POS_FRAMES)) != frame + 1):
                raise ValueError(f"could not decode native frame {frame}")
            yield tick, image
            cursor = frame + 1
    finally:
        capture.release()


def native_boxes(boxes, scores, width, height, detector_height):
    """Map global boxes to native pixels and apply the final stable 0.5 IoU NMS."""
    boxes = np.array(boxes, dtype=np.float32, copy=True)
    boxes /= np.array([1280 / width, detector_height / height] * 2, np.float32)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
    valid = np.all(boxes[:, 2:] > boxes[:, :2], axis=1)
    boxes = boxes[valid]
    order = np.argsort(-scores[valid], kind="stable")
    keep = []
    areas = (boxes[:, 2:] - boxes[:, :2]).prod(axis=1)
    while len(order):
        first, rest = order[0], order[1:]
        keep.append(first)
        overlap = np.maximum(0, np.minimum(boxes[first, 2:], boxes[rest, 2:])
                             - np.maximum(boxes[first, :2], boxes[rest, :2])).prod(axis=1)
        iou = overlap / np.maximum(areas[first] + areas[rest] - overlap, 1e-12)
        order = rest[iou <= .5]
    return boxes[np.array(keep, np.int64)]


class BallDetector:
    def __init__(self, weights):
        with weights.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != WEIGHTS_SHA256:
                raise ValueError("unexpected football-ball-detection.pt SHA256")
        for package, expected in RUNTIME.items():
            if metadata.version(package) != expected:
                raise ValueError(f"requires {package}=={expected}")

        import cv2
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("the tactical detector requires CUDA device 0")
        torch.set_num_threads(2)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        cv2.ocl.setUseOpenCL(False)
        cv2.setNumThreads(2)

        from ultralytics import YOLO

        self.model = YOLO(str(weights), task="detect")
        self.model.model.requires_grad_(False).float().eval()
        if {int(key): value.lower() for key, value in self.model.names.items()} != {0: "ball"}:
            raise ValueError("expected a single ball class in the detector checkpoint")

    def predict(self, image):
        import cv2
        import torch

        height, width = image.shape[:2]
        detector_height = max(1, round(height * 1280 / width))
        pixels = cv2.resize(image, (1280, detector_height), interpolation=cv2.INTER_AREA)
        with torch.inference_mode():
            results = self.model.predict(source=pixels, imgsz=1280, conf=.25, iou=.7, max_det=300,
                                         batch=1, quantize=None, augment=False, rect=False, save=False,
                                         verbose=False, device="0", stream=False, classes=None)
        if len(results) != 1 or self.model.predictor.device.type != "cuda":
            raise RuntimeError("expected a single CUDA detector result")
        output = results[0].boxes
        boxes = output.xyxy.detach().cpu().numpy().astype(np.float32)
        scores = output.conf.detach().cpu().numpy().astype(np.float32)
        if (torch.any(output.cls != 0).item() or not np.isfinite(boxes).all()
                or not np.isfinite(scores).all() or (scores < .25).any()
                or (scores > 1).any() or (boxes[:, 2:] < boxes[:, :2]).any()):
            raise ValueError("unexpected detector output")
        return native_boxes(boxes, scores, width, height, detector_height)
