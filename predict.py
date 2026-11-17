"""Reproduce the tiki data possession CSV from the official tactical match bundle."""
import argparse
import csv
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from urllib.request import urlopen
from importlib import metadata

ROOT = Path(__file__).resolve().parent
DETECTOR_URL = ("https://drive.usercontent.google.com/download?"
                "id=1isw4wx-MK9h9LMr36VvIWlJD6ppUvw7V&export=download&confirm=t")
DETECTOR_SIZE = 136_786_099
DETECTOR_SHA = "678fbad05134f19c5094cb8d273812ec9c6691228180d46832551ecf99ed2912"
BASE_SHA = "b8de956db3907c1da6a808bffcc57227e0a23be4302208d21711ab77ed687469"
COORD_SHA = "49a32b97805a79a241c359512cd5f3ba8996246b67442c57d1b3483f6fb75a80"
VIDEO_SHA = "7cad6d1b044aa416c49f78ca865b1856b6b2099f04c544c1d27e37975d97d34e"
FINAL_SHA = "26105d533b226b95e85d29aad0809097171d9e70684521c888c8501dc9a5e6ae"
FRAMES = 199319

# Six coordinate checkpoints: trained on the first half of J03WOH from the
# DFL/Bassek et al. (2025) dataset, https://doi.org/10.6084/m9.figshare.28196177.v1
# (CC BY 4.0). Training used a first-half XML subset with closing tags appended.
# The separately distributed detector is the Roboflow Sports soccer checkpoint:
# https://github.com/roboflow/sports/blob/42c80c06b6b65a7f89455b89fe31cdf4c38ba227/examples/soccer/setup.sh

def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

def fetch_detector(dest):
    if dest.exists():
        if sha(dest) != DETECTOR_SHA:
            raise ValueError(f"Existing detector file has the wrong digest: {dest}")
        return
    dest.parent.mkdir(exist_ok=True, parents=True)
    with tempfile.TemporaryDirectory(dir=dest.parent) as folder:
        tmp = Path(folder) / "ball.pt"
        total = 0
        with urlopen(DETECTOR_URL, timeout=90) as remote, tmp.open("xb") as f:
            while block := remote.read(1 << 20):
                total += len(block)
                if total > DETECTOR_SIZE:
                    raise ValueError("Unexpected detector download size")
                f.write(block)
        if total != DETECTOR_SIZE or sha(tmp) != DETECTOR_SHA:
            raise ValueError("Detector download failed its size or SHA256 check")
        tmp.replace(dest)

def as_csv(labels):
    if len(labels) != FRAMES or any(k not in ("H","A","D","X") for k in labels):
        raise ValueError("Expected one H/A/D/X label per tactical frame")
    return ("frame,label\n" + "".join(f"{i},{v}\n" for i,v in enumerate(labels))).encode("ascii")

def verify_bytes(data, expected, stage):
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise ValueError(f"{stage} differs from the submitted recipe: {actual}")

def preflight(data):
    inputs = ["BAR-ATH_tactical" + s for s in (".mp4","_atd.xml","_atd.txt","_homography.csv")]
    missing = [name for name in inputs if not (data / "tactical" / name).is_file()]
    if missing:
        raise ValueError("Missing official tactical inputs: " + ", ".join(missing))
    if sys.version_info[:3] != (3,13,12):
        raise ValueError("Use CPython 3.13.12 with the pinned requirements.txt")
    for line in (ROOT/"requirements.txt").read_text().splitlines():
        if not line.strip() or line.startswith(("#","--")):
            continue
        package, version = line.strip().split("==")
        if metadata.version(package) != version:
            raise ValueError(f"Install the pinned {package}=={version} before inference")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the submitted tactical-video detector")

def publish(data, temp, dest):
    """Write completely before publishing; do not overwrite an existing file."""
    ready = temp / "final.csv"
    with ready.open("xb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    if os.name == "nt":
        os.rename(ready, dest)  # Windows rename fails if dest exists.
    else:
        os.link(ready, dest)  # Same filesystem; link fails if dest exists.

def combine(base, coordinates, video):
    if len(base) != len(coordinates) or len(base) != len(video):
        raise ValueError("Frame counts do not match")
    result = []
    for b,c,v in zip(base, coordinates, video):
        if b in ("D","X") and (v != b or c != b):
            raise ValueError("Protected dead/outside-half baseline changed")
        if b in ("H","A") and (c == "X" or v not in ("H","A")):
            raise ValueError("Unexpected abstention or video status change")
        result.append(c if c == "D" or v == b else v)
    return result

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=r"""Setup (Windows x64, Python 3.13.12, NVIDIA CUDA):
  py -3.13 -m venv .venv
  .venv\Scripts\python -m pip install --no-deps -r requirements.txt
  .venv\Scripts\python predict.py --fetch-ball-model
  .venv\Scripts\python predict.py --data-dir D:/match/data --output tiki_data.csv

The data directory must contain tactical/BAR-ATH_tactical.mp4 and the matching
_atd.txt, _atd.xml and _homography.csv. Provide the official licensed bundle.
The included weights/ folder supplies the coordinate models. The detector can
be fetched from its upstream publisher or supplied with --ball-model.""")
    parser.add_argument("--data-dir", type=Path, help="Root of the official data bundle")
    parser.add_argument("--ball-model", type=Path, default=ROOT/"football-ball-detection.pt")
    parser.add_argument("--output", type=Path, default=Path("tiki_data.csv"))
    parser.add_argument("--fetch-ball-model", action="store_true", help="Download and verify the detector, then exit")
    args = parser.parse_args()
    if args.fetch_ball_model:
        fetch_detector(args.ball_model)
        print(f"Verified {args.ball_model}")
        return
    if args.data_dir is None:
        parser.error("supply --data-dir, or use --fetch-ball-model")
    if args.output.exists():
        parser.error("output already exists; use a new filename")
    if not args.ball_model.is_file() or sha(args.ball_model) != DETECTOR_SHA:
        parser.error("detector absent or changed: use --fetch-ball-model or --ball-model PATH")
    for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ[key]="2"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    preflight(args.data_dir)
    # Import numerical packages only after threading/runtime setup.
    from method.coordinate import run as coordinates
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="possession-", dir=args.output.parent) as scratch:
        tmp = Path(scratch)
        print("Predicting possession from full and half-pitch tracking views ...", flush=True)
        base, corrected = coordinates(args.data_dir.resolve(), ROOT/"weights", device="cuda")
        baseline = as_csv(base)
        verify_bytes(baseline, BASE_SHA, "Tracking baseline")
        verify_bytes(as_csv(corrected), COORD_SHA, "Coordinate corrections")
        (tmp/"base.csv").write_bytes(baseline)
        print("Checking tactical video for ball evidence ...", flush=True)
        # Use a fresh process so that the coordinate GRUs' deterministic Torch
        # settings do not alter the detector's original inference environment.
        subprocess.run([sys.executable, "-B", "-m", "video.overlay",
                        "--data-dir", str(args.data_dir.resolve()),
                        "--backbone", str((tmp/"base.csv").resolve()),
                        "--weights", str(args.ball_model.resolve()),
                        "--out", str((tmp/"video").resolve())], cwd=ROOT, check=True)
        path = tmp/"video/gap_recovery.csv"
        if sha(path) != VIDEO_SHA:
            raise ValueError("Tactical video output differs from the submitted recipe")
        with path.open(encoding="ascii", newline="") as f:
            visual = [row["label"] for row in csv.DictReader(f)]
        result = as_csv(combine(base, corrected, visual))
        verify_bytes(result, FINAL_SHA, "Final CSV")
        publish(result, tmp, args.output)
    print(f"{args.output}: {FRAMES} rows, SHA256 {FINAL_SHA}")

if __name__ == "__main__":
    main()
