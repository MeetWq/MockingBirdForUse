import torch
from typing import Optional

from . import hparams
from .models.fatchord_version import WaveRNN
from ...log import logger

_model: Optional[WaveRNN] = None
_device: Optional[torch.device] = None


def load_model(weights_fpath):
    global _model, _device

    logger.debug("Building Wave-RNN")
    _model = WaveRNN(
        rnn_dims=hparams.voc_rnn_dims,
        fc_dims=hparams.voc_fc_dims,
        bits=hparams.bits,
        pad=hparams.voc_pad,
        upsample_factors=hparams.voc_upsample_factors,
        feat_dims=hparams.num_mels,
        compute_dims=hparams.voc_compute_dims,
        res_out_dims=hparams.voc_res_out_dims,
        res_blocks=hparams.voc_res_blocks,
        hop_length=hparams.hop_length,
        sample_rate=hparams.sample_rate,
        mode=hparams.voc_mode,
    )

    if torch.cuda.is_available():
        _model = _model.cuda()
        _device = torch.device("cuda")
    else:
        _device = torch.device("cpu")

    logger.debug("Loading model weights at %s" % weights_fpath)
    checkpoint = torch.load(weights_fpath, _device)
    _model.load_state_dict(checkpoint["model_state"])
    _model.eval()


def is_loaded():
    return _model is not None


def infer_waveform(
    mel, normalize=True, batched=True, target=8000, overlap=800, progress_callback=None
):
    """
    Infers the waveform of a mel spectrogram output by the synthesizer (the format must match
    that of the synthesizer!)

    :param normalize:
    :param batched:
    :param target:
    :param overlap:
    :return:
    """
    if _model is None:
        raise Exception("Please load Wave-RNN in memory before using it")

    if normalize:
        mel = mel / hparams.mel_max_abs_value
    mel = torch.from_numpy(mel[None, ...])
    wav = _model.generate(
        mel, batched, target, overlap, hparams.mu_law, progress_callback
    )
    return wav, hparams.sample_rate
