import os
import torch
from pathlib import Path
from typing import Optional

from . import hparams
from .models import Generator
from ...log import logger

generator: Optional[Generator] = None
output_sample_rate: Optional[int] = None
_device: Optional[torch.device] = None


def check_device():
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    logger.debug("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    logger.debug("Complete.")
    return checkpoint_dict


def load_model(weights_fpath: Path):
    global generator, _device, output_sample_rate

    output_sample_rate = hparams.sampling_rate
    torch.manual_seed(hparams.seed)

    check_device()
    generator = Generator(hparams).to(_device)
    state_dict_g = load_checkpoint(weights_fpath, _device)
    generator.load_state_dict(state_dict_g["generator"])
    generator.eval()
    generator.remove_weight_norm()


def is_loaded():
    return generator is not None


def infer_waveform(mel):

    if generator is None:
        raise Exception("Please load hifi-gan in memory before using it")

    mel = torch.FloatTensor(mel).to(_device)
    mel = mel.unsqueeze(0)

    with torch.no_grad():
        y_g_hat = generator(mel)
        audio = y_g_hat.squeeze()
    audio = audio.cpu().numpy()

    return audio, output_sample_rate
