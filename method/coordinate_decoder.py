"""Decode 5 Hz state, ownership and directional event predictions at native frame rate."""

import numpy as np


def runs(mask):
    """Return the inclusive start and exclusive stop of each True interval."""
    return zip(np.flatnonzero(mask & np.r_[True, ~mask[:-1]]),
               np.flatnonzero(mask & np.r_[~mask[1:], True]) + 1)


def event_peaks(q, eligible):
    """First local maximum of a directional event posterior above its threshold."""
    direction = q[:, 1:].argmax(1) + 1
    peaks = {}
    for t in range(1, len(q)):
        k = direction[t]
        following = q[t + 1, k] if t + 1 < len(q) else -np.inf
        if eligible[t] and q[t, k] >= 0.5 and q[t, k] > q[t, 0] and q[t, k] > q[t - 1, k] and q[t, k] >= following:
            peaks[t] = int(k - 1)  # Sender: 0 for H→A, 1 for A→H.
    return peaks


def decode_run(labels, raw, events, peaks, blocks, emission_prefix, agreement_prefix, start, stop):
    """Optimize a hidden-ball region with eight states (owner × two reward-claim bits).

    The owner can change at a state-model boundary, at an ownership-model
    boundary in its predicted direction, or at a directional event peak.
    A peak can pay its reward once per direction within an event-dominance block.
    """
    raw_cuts = set(np.flatnonzero(raw[start + 1:stop] != raw[start:stop - 1]) + start + 1)
    owner_cuts = set(np.flatnonzero(labels[start + 1:stop] != labels[start:stop - 1]) + start + 1)
    allowed_peaks = {t: sender for t, sender in peaks.items() if start < t < stop - 1}
    ticks = sorted({start, stop, *raw_cuts, *owner_cuts, *allowed_peaks})
    back = np.full((len(ticks) - 1, 8), -1, np.int8)
    values = np.full(8, -np.inf)
    agrees = np.zeros(8, int)
    switches = np.zeros(8, int)
    ranks = np.arange(8)
    for j, (left, right) in enumerate(zip(ticks, ticks[1:])):
        emission = 0.2 * (emission_prefix[right] - emission_prefix[left])
        matched = agreement_prefix[right] - agreement_prefix[left]
        next_values = np.full(8, -np.inf)
        next_agrees = np.zeros(8, int)
        next_switches = np.zeros(8, int)
        rank_keys = [None] * 8
        if j == 0:
            for owner in range(2):
                state = owner * 4
                next_values[state] = emission[owner]
                next_agrees[state] = matched[owner]
                rank_keys[state] = (owner, 0)
        else:
            keys = [None] * 8
            sender_at_peak = allowed_peaks.get(left)
            reward = 0.0
            if sender_at_peak is not None:
                event_probability = max(float(events[left, sender_at_peak + 1]), 1e-9)
                no_event = max(float(events[left, 0]), 1e-9)
                reward = max(0.0, float(np.log(event_probability / no_event)))
            for prior in range(8):
                if not np.isfinite(values[prior]):
                    continue
                sender, bits = divmod(prior, 4)
                if blocks[left] != blocks[ticks[j - 1]]:
                    bits = 0
                for owner in range(2):
                    owner_edge = left in owner_cuts and labels[left] == "HA"[owner]
                    if sender != owner and left not in raw_cuts and sender != sender_at_peak and not owner_edge:
                        continue
                    options = [(bits, 0.0)]
                    bit = 1 << sender
                    if sender != owner and sender == sender_at_peak and reward > 0 and not bits & bit:
                        options.append((bits | bit, reward))
                    for mask, bonus in options:
                        state = owner * 4 + mask
                        # Ties prefer state agreement, then fewer switches, then predecessor order.
                        key = (float(values[prior] + bonus + emission[owner]), int(agrees[prior] + matched[owner]),
                               int(switches[prior]) - int(sender != owner), -sender, -int(ranks[prior]), -prior)
                        if keys[state] is None or key > keys[state]:
                            keys[state] = key
                            next_values[state], next_agrees[state], next_switches[state] = key[:3]
                            back[j, state] = prior
                            rank_keys[state] = (owner, int(ranks[prior]))
        order = {key: i for i, key in enumerate(sorted({key for key in rank_keys if key is not None}))}
        ranks = np.array([order.get(key, 8) for key in rank_keys])
        values, agrees, switches = next_values, next_agrees, next_switches
    state = max(range(8), key=lambda s: (values[s], int(agrees[s]), int(switches[s]),
                                        -(s // 4), -int(ranks[s]), -(s % 4)))
    for j in range(len(ticks) - 2, -1, -1):
        labels[ticks[j]:ticks[j + 1]] = "HA"[state // 4]
        state = int(back[j, state])


def decode(state, owner, events, ball_seen, counts, native_frames):
    codes = state.argmax(axis=-1)
    tied = (state == state.max(axis=-1, keepdims=True)).sum(axis=-1) > 1
    raw = np.array(list("HADX"))[np.where(tied, 3, codes)]
    live = np.isin(raw, ("H", "A"))
    team = np.where(owner[:, 0] > owner[:, 1], "H", np.where(owner[:, 1] > owner[:, 0], "A", "X"))
    labels = np.where(live, team, raw)
    if np.any(live & (team == "X")):
        raise ValueError("Owner posterior ties on a live frame")
    uncertain = state.max(1) < 2 / 3
    persistent = np.zeros(len(raw), bool)
    for a, b in runs(uncertain):
        if b - a > 50:
            persistent[a:b] = True
    eligible = persistent & live & ~ball_seen & (counts.min(1) >= 3)
    peaks = event_peaks(events, eligible)
    kind = events[:, 1:].argmax(axis=1) + 1
    kind[events[np.arange(len(events)), kind] <= events[:, 0]] = 0
    blocks = np.r_[0, np.cumsum(kind[1:] != kind[:-1])]
    log_probs = np.log(np.maximum(owner[:, :2].astype(np.float64), 1e-9))
    emission_prefix = np.vstack([np.zeros(2), np.cumsum(log_probs, axis=0)])
    agreement_prefix = np.vstack([np.zeros(2, dtype=int), np.cumsum(np.column_stack([raw == "H", raw == "A"]), axis=0)])
    for start, stop in runs(eligible):
        decode_run(labels, raw, events, peaks, blocks, emission_prefix, agreement_prefix, start, stop)
    return np.repeat(labels, 5)[:native_frames]
