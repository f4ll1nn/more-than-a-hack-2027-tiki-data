"""Apply the CUDA tactical-ball overlay to the full-player coordinate prediction."""

import argparse
import csv
import os
from pathlib import Path
from uuid import uuid4

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "2"
os.environ.update(OPENCV_OPENCL_RUNTIME="disabled", ULTRALYTICS_SAFE_LOAD="1",
                  TORCH_FORCE_WEIGHTS_ONLY_LOAD="1", YOLO_AUTOINSTALL="false", YOLO_OFFLINE="true")

import numpy as np

from .detector import BallDetector, read_frames
from .tracking import FRAME_COUNT, TacticalTracking


def read_backbone(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = csv.reader(stream)
        if next(rows, None) != ["frame", "label"]:
            raise ValueError("expected frame,label header")
        labels = []
        for frame, row in enumerate(rows):
            if len(row) != 2 or row[0] != str(frame) or row[1] not in ("H", "A", "D", "X"):
                raise ValueError(f"invalid backbone row at frame {frame}")
            labels.append(row[1])
    if len(labels) != FRAME_COUNT:
        raise ValueError(f"expected {FRAME_COUNT} backbone frames")
    return np.array(labels, dtype="U1")


def decode(backbone, tracking, contacts, allow_missing):
    """Require three distinct 5 Hz confirmations and cancel on native-frame conflict."""
    held = np.repeat(contacts, 5)[:len(backbone)]
    admitted = tracking.ball_observed & (tracking.coordinate_contact == held)
    if allow_missing:
        admitted |= ~tracking.ball_observed
    live = np.isin(backbone, ["H", "A"])
    eligible = tracking.available & live & admitted
    prediction = backbone.copy()
    pending, count = -1, 0
    for frame, usable in enumerate(eligible):
        if not usable:
            pending, count = -1, 0
            continue
        if frame % 5 == 0:
            candidate = int(contacts[frame // 5])
            if candidate < 0:
                pending, count = -1, 0
            else:
                count = count + 1 if candidate == pending else 1
                pending = candidate
        if count >= 3:
            prediction[frame] = "HA"[pending]
    return prediction


def possible_windows(backbone, tracking, allow_missing):
    """Probe constant H and A contacts to find every three-tick window that could matter."""
    windows = []
    for owner in (0, 1):
        contacts = np.full((len(backbone) + 4) // 5, owner, np.int8)
        possible = decode(backbone, tracking, contacts, allow_missing)
        for end in np.unique(np.flatnonzero(possible != backbone) // 5):
            windows.append((int(end), owner))
    return windows


def next_observations(windows, known):
    """Choose shared unknown ticks from windows with no contradictory observation."""
    memberships = {}
    for index, (end, owner) in enumerate(windows):
        ticks = range(end - 2, end + 1)
        if any(tick in known and known[tick] != owner for tick in ticks):
            continue
        for tick in ticks:
            if tick not in known:
                memberships.setdefault(tick, set()).add(index)
    remaining = set().union(*memberships.values())
    chosen = []
    while remaining:
        best = min(memberships, key=lambda tick: (-len(memberships[tick] & remaining), tick))
        chosen.append(best)
        remaining -= memberships.pop(best)
    return sorted(chosen)


def predict(backbone, tracking, detector):
    known = {}
    # Match the original order: resolve fully corroborated windows, then all
    # other windows where an absent coordinate ball permits a visual correction.
    for allow_missing in (False, True):
        windows = possible_windows(backbone, tracking, allow_missing)
        while ticks := next_observations(windows, known):
            print(f"Decoding {len(ticks)} tactical observations", flush=True)
            for tick, image in read_frames(tracking.video_path, ticks, len(backbone)):
                known[tick] = tracking.video_contact(tick * 5, detector.predict(image))

    # Any remaining unknown tick is irrelevant to the full native result.
    predictions = []
    for fill in (-1, 0, 1):
        contacts = np.full((len(backbone) + 4) // 5, fill, np.int8)
        for tick, owner in known.items():
            contacts[tick] = owner
        predictions.append(decode(backbone, tracking, contacts, allow_missing=True))
    if any(not np.array_equal(predictions[0], other) for other in predictions[1:]):
        raise ValueError("sparse observations leave undetermined native predictions")
    print(f"Resolved {len(backbone)} frames with {len(known)} detector observations", flush=True)
    return predictions[0]


def run(data_dir: Path, backbone_path: Path, weights: Path, out: Path) -> Path:
    """Write gap_recovery.csv to the output directory and return its path."""
    out = Path(out)
    output = out / "gap_recovery.csv"
    if output.exists():
        raise FileExistsError(output)
    backbone = read_backbone(backbone_path)
    runtime = (out / "runtime").resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    os.environ.update(YOLO_CONFIG_DIR=str(runtime), MPLCONFIGDIR=str(runtime))
    detector = BallDetector(Path(weights))
    tracking = TacticalTracking.load(data_dir)
    labels = predict(backbone, tracking, detector)
    staged = out / f".{uuid4().hex}.csv.tmp"
    try:
        with staged.open("w", encoding="utf-8", newline="") as stream:
            stream.write("frame,label\n")
            stream.writelines(f"{frame},{label}\n" for frame, label in enumerate(labels))
        staged.replace(output)
    finally:
        staged.unlink(missing_ok=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="root containing tactical/")
    parser.add_argument("--backbone", type=Path, required=True, help="full-player coordinate CSV")
    parser.add_argument("--weights", type=Path, required=True, help="football-ball-detection.pt")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    args = parser.parse_args()
    print(run(args.data_dir, args.backbone, args.weights, args.out))
