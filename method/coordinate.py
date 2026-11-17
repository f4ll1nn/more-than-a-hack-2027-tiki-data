"""Possession from tactical tracking at 25 Hz, using six frozen models."""

from dataclasses import dataclass, replace
from hashlib import file_digest
import os
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from .coordinate_features import features
from .coordinate_decoder import decode


# Inclusive start and exclusive stop on the tactical video timeline.
HALVES = ((11185, 90652), (110808, 190198))
FRAME_COUNT = 199319
WEIGHT_HASHES = {
    "state17": "90e6abf380da4060c4380cbcf8f33057eeec11d2f77f7b75ae3764ca5efa872c",
    "state29": "5e42ed5a45b9ddf4e08eaed45ac26c462e33c410c2d25a5e6e32c722daccf2cb",
    "event17": "aeb57e95b7d2fbfc1dec8e4a1be2e9e0b093d862cd7cbd4e653de8adc84f470e",
    "event29": "c44741195a9dc8d2e14c23a14618fbd071920c268c74741b3c649bb1f612fa4a",
    "owner17": "197cf790fc3b47b4aec1852db460389fd4c1fa57e951fd8fd225f46134c98948",
    "owner29": "24738396c8c79b424d8c60361cb3f133ce4a39a71029cfecafeac0ace8c3c773",
}


@dataclass(frozen=True)
class Tracking:
    pitch_m: tuple[float, float]
    player_team: np.ndarray
    player_xy_m: np.ndarray
    player_valid: np.ndarray
    ball_xy_m: np.ndarray
    ball_valid: np.ndarray

    def half(self, start, stop):
        return replace(self, player_xy_m=self.player_xy_m[start:stop],
                       player_valid=self.player_valid[start:stop],
                       ball_xy_m=self.ball_xy_m[start:stop], ball_valid=self.ball_valid[start:stop])

    def view(self, side):
        if side == "full":
            return self
        x = self.player_xy_m[:, :, 0]
        midpoint = self.pitch_m[0] / 2
        if side == "left":
            keep = self.player_valid & (x <= midpoint)
        elif side == "right":
            keep = self.player_valid & (x >= midpoint)
        else:
            raise ValueError(f"Unknown player view {side!r}")
        return replace(self, player_xy_m=np.where(keep[..., None], self.player_xy_m, np.nan),
                       player_valid=keep)


def load_tracking(data_dir: Path) -> Tracking:
    """Read the official ATD slot order and convert normalized positions to metres."""
    base = Path(data_dir) / "tactical" / "BAR-ATH_tactical_atd"
    root = ElementTree.parse(base.with_suffix(".xml")).getroot()
    for node in root.iter():
        node.tag = node.tag.rsplit("}", 1)[-1]
    teams = [node.get("id") for node in root.iter("Team")]
    members = [node.get("teamId") for node in root.iter("Player")]
    if len(teams) != 2 or None in teams or teams[0] == teams[1] or set(members) != set(teams):
        raise ValueError("Expected two distinct teams with assigned players")
    player_team = np.array([teams.index(team) for team in members], dtype=np.int8)
    def dimension(name, fallback):
        value = root.findtext(".//" + name)
        if value is None:
            value = root.findtext(".//FieldSize/" + fallback)
        size = float(value)
        if not np.isfinite(size) or size <= 0:
            raise ValueError(f"Invalid pitch {name}")
        return size
    pitch = (dimension("PitchLength", "Width"), dimension("PitchWidth", "Height"))
    players = np.empty((FRAME_COUNT, len(members), 3))
    ball = np.empty((FRAME_COUNT, 3))
    count = 0
    with base.with_suffix(".txt").open(encoding="utf-8-sig") as stream:
        for count, line in enumerate(stream, 1):
            frame, player_fields, ball_fields = line.strip().split(":")
            if count > FRAME_COUNT or int(frame) != count:
                raise ValueError(f"Invalid tracking frame at line {count}")
            p = np.array([[float(v) for v in group.split(",")] for group in player_fields.split(";")])
            b = np.array([float(v) for v in ball_fields.split(",")])
            if p.shape != (len(members), 3) or b.shape != (3,) or np.isinf(p).any() or np.isinf(b).any():
                raise ValueError(f"Invalid coordinates at line {count}")
            players[count - 1] = p
            ball[count - 1] = b
    if count != FRAME_COUNT:
        raise ValueError(f"Expected {FRAME_COUNT} tracking frames, received {count}")
    player_xy = players[..., :2]
    ball_xy = ball[..., :2]
    player_xy *= pitch
    ball_xy *= pitch
    return Tracking(pitch, player_team, player_xy, np.isfinite(player_xy).all(-1),
                    ball_xy, np.isfinite(ball_xy).all(-1))


def predict_views(tracking, weights_dir: Path, device: str = "cpu"):
    """Yield (view, native labels) for the full, left and right player views."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from .coordinate_models import load_model, posteriors

    if device not in ("cpu", "cuda"):
        raise ValueError("Expected coordinate device cpu or cuda")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    models = {}
    for name, digest in WEIGHT_HASHES.items():
        path = Path(weights_dir) / (name + ".pt")
        with path.open("rb") as stream:
            if file_digest(stream, "sha256").hexdigest() != digest:
                raise ValueError(f"Unexpected checkpoint bytes in {path}")
        models[name] = load_model(name[:-2], path, device)
    for side in ("full", "left", "right"):
        torch.set_num_threads(4 if side == "full" else 2)
        labels = np.full(len(tracking.ball_valid), "X", dtype="U1")
        for start, stop in HALVES:
            print(f"Coordinate {side}, tactical frames [{start}, {stop})", flush=True)
            half = tracking.half(start, stop).view(side)
            state_features, entities, counts = features(half)
            state, owner, events = posteriors(models, state_features, entities, counts, device)
            labels[start:stop] = decode(state, owner, events, half.ball_valid[::5], counts, stop - start)
        yield side, labels


def correct_coordinates(tracking, base, left, right):
    """Recover supported short hidden gaps, then apply measured pitch exits."""
    hidden = base.copy()
    midpoint = tracking.pitch_m[0] / 2
    shared = (tracking.player_valid & (tracking.player_xy_m[:, :, 0] == midpoint)).any(axis=1)
    edges = np.r_[0, np.flatnonzero(base[1:] != base[:-1]) + 1, len(base)]
    previous_stop = -1
    for a, b in zip(edges[1:-2], edges[2:-1], strict=True):
        flank = base[a - 1]
        if (base[a] not in ("H", "A") or flank not in ("H", "A") or base[a] == flank
                or base[b] != flank or b - a >= 50 or a <= previous_stop):
            continue
        guard = slice(a - 1, b + 1)
        if tracking.ball_valid[a:b].any() or shared[guard].any():
            continue
        if (left[guard] == flank).all() and (right[guard] == flank).all():
            hidden[a:b] = flank
            previous_stop = b
    xy = tracking.ball_xy_m
    outside = tracking.ball_valid & ((xy < -0.11) | (xy > np.asarray(tracking.pitch_m) + 0.11)).any(axis=1)
    return np.where(outside & np.isin(hidden, ("H", "A")), "D", hidden)


def run(data_dir: Path, weights_dir: Path, device: str = "cpu"):
    """Return full-player and corrected coordinate labels on the tactical timeline."""
    tracking = load_tracking(data_dir)
    views = dict(predict_views(tracking, weights_dir, device))
    return views["full"], correct_coordinates(tracking, views["full"], views["left"], views["right"])
