import json
import os

import torch


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(model, path, device):
    model.load_state_dict(
        torch.load(path, map_location=device)
    )
    model.eval()
    return model


def save_history(history, path):
    with open(path, "w") as f:
        json.dump(history, f, indent=4)


def load_history(path):
    with open(path, "r") as f:
        return json.load(f)