"""Read aligned tactical tracking and turn ball locations into team observations."""

import csv
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


FRAME_COUNT = 199319
# Official tactical-video frame ranges, both endpoints inclusive.
HALVES = ((11185, 90651), (110808, 190197))


def read_metadata(path):
    root = ET.parse(path).getroot()
    for node in root.iter():
        node.tag = node.tag.rsplit("}", 1)[-1]
    teams = [node.get("id") for node in root.iter("Team")]
    members = [node.get("teamId") for node in root.iter("Player")]
    if len(teams) != 2 or None in teams or teams[0] == teams[1]:
        raise ValueError("expected two distinct metadata teams")
    if not members or any(team not in teams for team in members):
        raise ValueError("unknown or absent slot team assignments")

    def dimension(name, field):
        value = root.findtext(".//" + name)
        if value is None:
            value = root.findtext(".//FieldSize/" + field)
        if value is None or not np.isfinite(float(value)) or float(value) <= 0:
            raise ValueError("missing or invalid pitch dimension")
        return float(value)

    pitch = np.array([dimension("PitchLength", "Width"), dimension("PitchWidth", "Height")])
    return np.array([teams.index(team) for team in members], np.int8), pitch


def read_positions(path, slots, pitch):
    # The detector branch stores normalized player positions at float32 precision.
    players = np.empty((FRAME_COUNT, slots, 2), np.float32)
    ball = np.empty((FRAME_COUNT, 2), np.float64)
    with path.open(encoding="utf-8-sig") as stream:
        for frame in range(FRAME_COUNT):
            fields = stream.readline().rstrip("\r\n").split(":")
            if len(fields) != 3 or int(fields[0]) != frame + 1:
                raise ValueError(f"invalid ATD row for frame {frame}")
            slot_xy = [slot.split(",")[:2] for slot in fields[1].split(";")]
            if len(slot_xy) != slots:
                raise ValueError(f"unexpected ATD slot count for frame {frame}")
            players[frame] = slot_xy
            ball_fields = fields[2].split(",")
            if len(ball_fields) != 3:
                raise ValueError(f"invalid ATD ball at frame {frame}")
            ball[frame] = ball_fields[:2]
        if stream.readline():
            raise ValueError("additional ATD frame")
    if np.isinf(players).any() or np.isinf(ball).any() or np.any(np.isnan(ball[:, 0]) != np.isnan(ball[:, 1])):
        raise ValueError("invalid automatic tracking coordinates")
    return players, ball * pitch


def read_homographies(path):
    matrices = np.zeros((FRAME_COUNT, 3, 3), np.float64)
    valid = np.zeros(FRAME_COUNT, bool)
    keys = [f"h{i}{j}" for i in range(3) for j in range(3)]
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not set(["frame", "valid", *keys]).issubset(reader.fieldnames or ()):
            raise ValueError("missing tactical homography columns")
        for frame in range(FRAME_COUNT):
            row = next(reader, None)
            if row is None or int(row["frame"]) != frame or row["valid"] not in ("0", "1"):
                raise ValueError(f"invalid homography row at frame {frame}")
            if row["valid"] == "1":
                matrix = np.array([float(row[key]) for key in keys]).reshape(3, 3)
                if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
                    raise ValueError("singular or nonfinite homography")
                matrices[frame] = matrix
                valid[frame] = True
        if next(reader, None) is not None:
            raise ValueError("additional homography row")
    return matrices, valid


def nearest_team(ball_m, players_m, teams, pitch):
    """Require an in-pitch ball, both teams, a 2 m radius and a 1 m margin."""
    delta = players_m - ball_m[:, None, :]
    distance = np.linalg.norm(delta, axis=-1)
    distances = np.full((len(distance), 2), np.inf)
    for owner in (0, 1):
        team_distances = distance[:, teams == owner]
        distances[:, owner] = np.min(team_distances, axis=1, initial=np.inf)
    inside = ((ball_m >= 0) & (ball_m <= pitch)).all(axis=1)
    indices = np.flatnonzero(inside & np.isfinite(distances).all(axis=1))
    owner = distances[indices].argmin(axis=1)
    closest = distances[indices, owner]
    qualified = (closest <= 2.0) & (distances[indices, 1 - owner] - closest >= 1.0)
    result = np.full(len(ball_m), -1, np.int8)
    result[indices[qualified]] = owner[qualified]
    return result


@dataclass
class TacticalTracking:
    video_path: Path
    pitch: np.ndarray
    teams: np.ndarray
    players_m: np.ndarray
    homographies: np.ndarray
    available: np.ndarray
    ball_observed: np.ndarray
    coordinate_contact: np.ndarray

    @classmethod
    def load(cls, data_dir):
        folder = Path(data_dir) / "tactical"
        stem = "BAR-ATH_tactical"
        teams, pitch = read_metadata(folder / f"{stem}_atd.xml")
        players, ball_m = read_positions(folder / f"{stem}_atd.txt", len(teams), pitch)
        matrices, available = read_homographies(folder / f"{stem}_homography.csv")
        player_valid = ((players >= 0) & (players <= 1)).all(axis=2)
        players_m = players * pitch
        # An infinity placeholder leaves the other team's distance calculation intact.
        players_m[~player_valid] = np.inf
        for owner in (0, 1):
            available &= player_valid[:, teams == owner].any(axis=1)
        inside_half = np.zeros(FRAME_COUNT, bool)
        for first, last in HALVES:
            inside_half[first:last + 1] = True
        available &= inside_half

        contact = np.empty(FRAME_COUNT, np.int8)
        for start in range(0, FRAME_COUNT, 10000):
            chunk = slice(start, start + 10000)
            contact[chunk] = nearest_team(ball_m[chunk], players_m[chunk], teams, pitch)
        return cls(folder / f"{stem}.mp4", pitch, teams, players_m, matrices,
                   available, np.isfinite(ball_m).all(axis=1), contact)

    def video_contact(self, frame, boxes):
        if not len(boxes) or not self.available[frame]:
            return -1
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        homogeneous = np.column_stack([centers, np.ones(len(boxes))])
        normalized = homogeneous @ np.linalg.inv(self.homographies[frame]).T
        if not np.isfinite(normalized).all() or np.any(np.abs(normalized[:, 2]) <= 1e-12):
            return -1
        ball_m = normalized[:, :2] / normalized[:, 2, None] * self.pitch
        owners = nearest_team(ball_m, self.players_m[frame:frame + 1], self.teams, self.pitch)
        return int(owners[0]) if np.all(owners == owners[0]) else -1
