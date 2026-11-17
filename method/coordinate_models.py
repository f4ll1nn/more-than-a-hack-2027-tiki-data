"""Three paired recurrent models and the reflection/team-swap ensemble."""

import numpy as np
import torch
from torch import nn


class StateModel(nn.Module):
    """Opposite time-direction GRUs infer home, away or dead status."""

    def __init__(self):
        super().__init__()
        self.left = nn.GRU(32, 16, batch_first=True)
        self.right = nn.GRU(32, 16, batch_first=True)
        self.state = nn.Linear(32, 3)
        self.boundary = nn.Linear(32, 1)  # Auxiliary checkpoint head.

    def forward(self, x):
        width = x.shape[1]
        lengths = torch.full((len(x),), width, dtype=torch.long)

        def stream(layer, sequence):
            packed = nn.utils.rnn.pack_padded_sequence(sequence, lengths, batch_first=True, enforce_sorted=False)
            output, _ = layer(packed)
            return nn.utils.rnn.pad_packed_sequence(output, batch_first=True, total_length=width)[0]

        left = stream(self.left, x)
        reverse = torch.arange(width - 1, -1, -1, device=x.device).expand(len(x), -1)
        right = stream(self.right, x.gather(1, reverse[..., None].expand(-1, -1, 32)))
        right = right.gather(1, reverse[..., None].expand(-1, -1, 16))
        return self.state(torch.cat([left, right], -1))


class EventModel(nn.Module):
    """From player context, predict no event, home-to-away or away-to-home."""

    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(18, 128, num_layers=2, bidirectional=True, batch_first=True)
        self.head = nn.Linear(256, 3)

    def forward(self, x):
        output, _ = self.gru(x)
        return self.head(output)


class TeamAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(128, 384)
        self.out = nn.Linear(128, 128)
        self.norm1 = nn.LayerNorm(128)
        self.ff = nn.Sequential(nn.Linear(128, 256), nn.GELU(), nn.Linear(256, 128))
        self.norm2 = nn.LayerNorm(128)

    def forward(self, x, valid, weight, teams):
        batch, ticks, nodes, _ = x.shape
        q, k, v = self.qkv(x).reshape(batch, ticks, nodes, 3, 4, 32).permute(3, 0, 1, 4, 2, 5).unbind(0)
        scores = (q @ k.transpose(-1, -2)) / (32 ** 0.5) + weight.clamp(min=1e-9).log()[:, :, None, None, :]
        scores = scores.masked_fill(~valid[:, :, None, None, :], -10000.0)
        other_team = teams[:, :, None] != teams[:, None, :]
        scores = scores.masked_fill(other_team[:, None, None, :, :], -10000.0)
        attended = (scores.softmax(-1) @ v).transpose(2, 3).reshape(batch, ticks, nodes, 128)
        x = self.norm1(x + self.out(attended))
        return self.norm2(x + self.ff(x)) * valid[..., None]


class OwnerModel(nn.Module):
    """Pool same-team attention over player slots and decode an entire half."""

    def __init__(self):
        super().__init__()
        self.input = nn.Linear(13, 128)
        self.layers = nn.ModuleList([TeamAttention(), TeamAttention()])
        self.temporal = nn.GRU(512, 128, num_layers=2, bidirectional=True, batch_first=True)
        self.state = nn.Linear(256, 3)
        # These two auxiliary heads are included in the frozen state dictionary.
        self.near = nn.Sequential(nn.Linear(384, 64), nn.GELU(), nn.Linear(64, 1))
        self.none = nn.Linear(256, 1)

    def spatial(self, x, valid, weight):
        teams = x[:, 0, :, 11:13].argmax(-1)
        z = self.input(x) * valid[..., None]
        for layer in self.layers:
            z = layer(z, valid, weight, teams)
        pooled = []
        for team in (0, 1):
            selected = valid & (teams[:, None, :] == team)
            w = weight * selected
            mean = (z * w[..., None]).sum(2) / w.sum(2, keepdim=True).clamp(min=1)
            maximum = z.masked_fill(~selected[..., None], -10000.0).max(2).values
            maximum = torch.where(selected.any(2)[..., None], maximum, torch.zeros_like(maximum))
            pooled.extend([mean, maximum])
        return torch.cat(pooled, -1)

    def forward(self, pooled):
        output, _ = self.temporal(pooled)
        return self.state(output)


def load_model(kind, path, device):
    model = {"state": StateModel, "owner": OwnerModel, "event": EventModel}[kind]()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return model.to(device).eval()


def symmetric_state_or_event(model, x, device, *, state):
    if state:
        swap = [*range(5, 10), *range(0, 5), *range(10, 16), *range(24, 32), *range(16, 24)]
        reflect = [16, 17, 20, 21, 24, 25, 28, 29]
        undo = [1, 0, 2]
    else:
        swap = [*range(8, 16), *range(0, 8), 17, 16]
        reflect = [0, 1, 4, 5, 8, 9, 12, 13]
        undo = [0, 2, 1]
    reflected = x.copy()
    reflected[:, reflect] *= -1
    values = np.stack([x, x[:, swap], reflected, reflected[:, swap]])
    p = model(torch.from_numpy(values).to(device)).softmax(-1).cpu().numpy()
    return (p[0] + p[1][:, undo] + p[2] + p[3][:, undo]) / 4


def symmetric_owner(model, pack, device):
    x = pack["nodes"]
    valid = np.repeat(pack["valid"][None], 2, axis=0)
    weight = np.repeat(pack["weight"][None], 2, axis=0)
    result = None
    for reflect in (False, True):
        regular = x.copy() if reflect else x
        if reflect:
            regular[..., [0, 1, 2, 3, 5, 6]] *= -1
        swapped = regular.copy()
        swapped[..., 11:13] = regular[..., [12, 11]]
        values = np.stack([regular, swapped])
        # Compute spatial context in chunks. Both GRU layers still see the full half.
        pooled = []
        for a in range(0, len(x), 256):
            span = slice(a, a + 256)
            pooled.append(model.spatial(torch.from_numpy(values[:, span]).to(device),
                                        torch.from_numpy(valid[:, span]).to(device),
                                        torch.from_numpy(weight[:, span]).to(device)))
        state = model(torch.cat(pooled, 1)).softmax(-1).cpu().numpy()
        pair = state[0] + state[1][:, [1, 0, 2]]
        result = pair if result is None else result + pair
    return result / 4


@torch.inference_mode()
def posteriors(models, state_features, entities, counts, device):
    events = np.concatenate([state_features[:, 16:], np.minimum(counts, 11) / 11], axis=1).astype(np.float32)
    state = np.mean(np.stack([symmetric_state_or_event(models[f"state{s}"], state_features, device, state=True)
                              for s in (17, 29)]), axis=0)
    event = np.mean(np.stack([symmetric_state_or_event(models[f"event{s}"], events, device, state=False)
                              for s in (17, 29)]), axis=0)
    owner = np.mean(np.stack([symmetric_owner(models[f"owner{s}"], entities, device) for s in (17, 29)]), axis=0)
    return state, owner, event
