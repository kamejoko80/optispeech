import io
import os
import sys
import warnings
from importlib.util import find_spec
from math import ceil
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig

from . import pylogger, rich_utils

log = pylogger.get_pylogger(__name__)

matplotlib.use("Agg")


def extras(cfg: DictConfig) -> None:
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def task_wrapper(task_func: Callable) -> Callable:
    def wrap(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        try:
            metric_dict, object_dict = task_func(cfg=cfg)
        except Exception as ex:
            log.exception("")
            raise ex
        finally:
            log.info(f"Output dir: {cfg.paths.output_dir}")
            if find_spec("wandb"):
                import wandb

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()

        return metric_dict, object_dict

    return wrap


def get_metric_value(metric_dict: Dict[str, Any], metric_name: str) -> float:
    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise ValueError(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")
    return metric_value


def intersperse(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def save_figure_to_numpy(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape((h, w, 4))
    return buf[:, :, :3].copy()


def plot_attention(attn):
    fig = plt.figure(figsize=(12, 6))
    plt.imshow(attn.T, interpolation="nearest", aspect="auto")
    data = save_figure_to_numpy(fig)
    plt.close(fig)
    return data


def plot_tensor(tensor):
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(tensor, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    data = save_figure_to_numpy(fig)
    plt.close(fig)
    return data


def save_plot(tensor, savepath):
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(tensor, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close(fig)


def to_numpy(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    if isinstance(tensor, list):
        return np.array(tensor)
    raise TypeError("Unsupported type for conversion to numpy array")


def plot_spectrogram_to_numpy(spectrogram, filename):
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.title("Synthesised Mel-Spectrogram")
    plt.savefig(filename)
    plt.close(fig)


def get_phoneme_durations(durations, phones):
    prev = durations[0]
    merged_durations = []
    for i in range(1, len(durations), 2):
        if i == len(durations) - 2:
            next_half = durations[i + 1]
        else:
            next_half = ceil(durations[i + 1] / 2)

        curr = prev + durations[i] + next_half
        prev = durations[i + 1] - next_half
        merged_durations.append(curr)

    assert len(phones) == len(merged_durations)
    assert len(merged_durations) == (len(durations) - 1) // 2

    merged_durations = torch.cumsum(torch.tensor(merged_durations), 0, dtype=torch.long)
    start = torch.tensor(0)
    duration_json = []
    for i, duration in enumerate(merged_durations):
        duration_json.append(
            {
                phones[i]: {
                    "starttime": start.item(),
                    "endtime": duration.item(),
                    "duration": duration.item() - start.item(),
                }
            }
        )
        start = duration

    assert list(duration_json[-1].values())[0]["endtime"] == sum(
        durations
    ), f"{list(duration_json[-1].values())[0]['endtime'],  sum(durations)}"
    return duration_json


def get_model_size_mb(model):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    num_bytes = len(buf.getbuffer())
    return num_bytes // 1e6


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
