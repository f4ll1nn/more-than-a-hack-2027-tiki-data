"""Fixed tracking features with physical scales in metres and seconds."""


import warnings

import numpy as np


def _derived_speeds(obs):
    """Estimate speed over five native frames where both endpoints exist."""
    xy = obs.player_xy_m
    speed = np.full(xy.shape[:2], np.nan)
    good = obs.player_valid[5:] & obs.player_valid[:-5]
    value = np.linalg.norm(xy[5:] - xy[:-5], axis=-1) / 0.2
    speed[5:] = np.where(good, value, np.nan)
    return speed


def _base_features(obs):
    """Ball distance, team density, player flow, ball age and motion."""
    n = len(obs.ball_valid)
    ball, visible = (obs.ball_xy_m, obs.ball_valid)
    distance = np.linalg.norm(obs.player_xy_m - ball[:, None], axis=-1)
    distance[~obs.player_valid | ~np.isfinite(distance)] = np.inf
    speed = _derived_speeds(obs)
    columns = []
    for team in (0, 1):
        selected = obs.player_team == team
        d = np.sort(distance[:, selected], axis=1)
        if not d.shape[1]:
            raise ValueError('both teams required')
        nearest = np.clip(d[:, 0] / 30, 0, 1) * visible
        mean3 = np.clip(d[:, :3] / 30, 0, 1).mean(axis=1) * visible
        density = np.clip((d <= 3).sum(axis=1) / 11, 0, 1) * visible
        valid = np.clip(obs.player_valid[:, selected].sum(axis=1) / 11, 0, 1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            flow = np.nanmedian(speed[:, selected], axis=1)
        flow = np.clip(np.nan_to_num(flow, nan=0.0) / 12, 0, 1)
        columns.extend([nearest, mean3, density, valid, flow])
    index = np.arange(n)
    last = np.maximum.accumulate(np.where(visible, index, -1))
    age = np.where(last >= 0, (index - last) / 25.0, 20.0)
    columns.extend([visible.astype(float), np.clip(age / 20, 0, 1)])
    motion, flags = ([], [])
    prefix = np.r_[0, np.cumsum(~visible)]
    for lag in (5, 25):
        good = np.zeros(n, bool)
        if lag < n:
            good[lag:] = prefix[lag + 1:] - prefix[:n - lag] == 0
        value = np.zeros(n)
        delta = np.linalg.norm(ball[lag:] - ball[:-lag], axis=1) / (lag / 25.0)
        value[lag:] = np.where(good[lag:], np.clip(delta / 40, 0, 1), 0)
        motion.append(value)
        flags.append(good.astype(float))
    result = np.column_stack(columns + motion + flags).astype(np.float32)
    if result.shape != (n, 16) or not np.isfinite(result).all():
        raise ValueError('finite fixed feature matrix required')
    return result


def _player_bank(obs):
    """Deduplicate coincident same-team slots before measuring team shape and pressure."""
    n, teams, pitch = (len(obs.ball_valid), obs.player_team, np.array(obs.pitch_m))
    indices = np.arange(0, n, 5)
    out = np.zeros((len(indices), 16), np.float32)
    team_equal = teams[:, None] == teams[None, :]
    earlier = np.tri(len(teams), k=-1, dtype=bool)
    for begin in range(0, len(indices), 512):
        idx = indices[begin:begin + 512]
        xy = obs.player_xy_m[idx]
        valid = obs.player_valid[idx]
        equal = (xy[:, :, None, :] == xy[:, None, :, :]).all(axis=-1) & valid[:, :, None] & valid[:, None, :] & team_equal
        unique = valid & ~(equal & earlier).any(axis=2)
        past_idx = np.maximum(idx - 5, 0)
        past = obs.player_xy_m[past_idx]
        velocity = (xy - past) / 0.2
        velocity_valid = valid & obs.player_valid[past_idx] & (idx >= 5)[:, None]
        group_valid = equal & velocity_valid[:, None, :]
        minimum = np.where(group_valid[..., None], velocity[:, None, :, :], np.inf).min(axis=2)
        maximum = np.where(group_valid[..., None], velocity[:, None, :, :], -np.inf).max(axis=2)
        coherent = np.isfinite(minimum).all(axis=-1) & (minimum == maximum).all(axis=-1)
        group_velocity = np.where(coherent[..., None], minimum, np.nan)
        distance = np.linalg.norm(xy[:, :, None, :] - xy[:, None, :, :], axis=-1)
        distance[~valid[:, :, None] | ~valid[:, None, :]] = np.inf
        columns = []
        for team in (0, 1):
            selected = unique & (teams == team)
            count = selected.sum(axis=1)
            denominator = np.maximum(count, 1)
            center = np.where(selected[..., None], xy, 0).sum(axis=1) / denominator[:, None]
            spread = np.sqrt((np.where(selected[..., None], xy - center[:, None], 0) ** 2).sum(axis=1) / denominator[:, None])
            centered = np.where((count > 0)[:, None], np.clip(center / (pitch / 2) - 1, -1, 1), 0)
            spread = np.where((count > 0)[:, None], np.clip(spread / (pitch / 2), 0, 1), 0)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                flow = np.nanmedian(np.where(selected[..., None], group_velocity, np.nan), axis=1)
            flow = np.clip(np.nan_to_num(flow, nan=0.0) / 12, -1, 1)
            opponents = unique & (teams != team)
            nearest = np.where(opponents[:, None, :], distance, np.inf).min(axis=2)
            support = selected & np.isfinite(nearest)
            proximity = np.where(support, np.clip(nearest / 30, 0, 1), 0).sum(axis=1) / denominator
            pressure = (selected & (nearest <= 3)).sum(axis=1) / denominator
            columns.extend([centered[:, 0], centered[:, 1], spread[:, 0], spread[:, 1], flow[:, 0], flow[:, 1], proximity, pressure])
        out[begin:begin + len(idx)] = np.column_stack(columns)
    if not np.isfinite(out).all():
        raise ValueError('finite player bank required')
    return out


def _entity_features(obs):
    """Build thirteen features per player and two reserved team/ball nodes."""
    idx = np.arange(0, len(obs.ball_valid), 5)
    xy = obs.player_xy_m[idx]
    valid = obs.player_valid[idx]
    old = np.maximum(idx - 5, 0)
    vvalid = valid & obs.player_valid[old] & (idx >= 5)[:, None]
    velocity = np.where(vvalid[..., None], (xy - obs.player_xy_m[old]) / 0.2, 0)
    half = np.asarray(obs.pitch_m) / 2
    ball = obs.ball_xy_m[idx]
    observed = obs.ball_valid[idx]
    nodes = np.zeros((len(idx), xy.shape[1] + 2, 13), np.float32)
    nodes[:, :-2, :2] = np.where(valid[..., None], np.clip(xy / half - 1, -1, 1), 0)
    nodes[:, :-2, 2:4] = np.clip(np.nan_to_num(velocity, nan=0.0) / 12, -1, 1)
    nodes[:, :-2, 4] = vvalid
    relative = xy - ball[:, None, :]
    available = valid & observed[:, None]
    nodes[:, :-2, 5:7] = np.where(available[..., None], np.clip(relative / 30, -1, 1), 0)
    nodes[:, :-2, 7] = np.where(available, np.clip(np.linalg.norm(relative, axis=-1) / 30, 0, 1), 0)
    nodes[:, :, 8] = observed[:, None]
    nodes[:, :-2, 9] = valid
    nodes[:, -2:, 9:11] = 1
    teams = np.r_[obs.player_team, 0, 1]
    nodes[:, :, 11] = teams == 0
    nodes[:, :, 12] = teams == 1
    nodes[:, -2:, :2] = np.where(observed[:, None, None], np.clip(ball[:, None, :] / half - 1, -1, 1), 0)
    mask = np.column_stack([valid, np.ones((len(idx), 2), bool)])
    same = (xy[:, :, None, :] == xy[:, None, :, :]).all(-1) & valid[:, :, None] & valid[:, None, :] & (obs.player_team[:, None] == obs.player_team[None, :])
    weight = np.column_stack([np.where(valid, 1 / np.maximum(same.sum(-1), 1), 0), np.ones((len(idx), 2))]).astype(np.float32)
    if not np.isfinite(nodes).all():
        raise ValueError('finite masked entity inputs required')
    return {'nodes': nodes, 'valid': mask, 'weight': weight}


def features(obs):
    """Return the state features, masked player nodes and visible team counts at 5 Hz."""
    state = np.concatenate([_base_features(obs)[::5], _player_bank(obs)], axis=1).astype(np.float32)
    entity = _entity_features(obs)
    counts = np.column_stack([obs.player_valid[::5, obs.player_team == team].sum(1) for team in (0, 1)])
    return (state, entity, counts)
