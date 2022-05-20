import re
import librosa
import numpy as np
from io import BytesIO
from pathlib import Path
from scipy.io import wavfile
from typing import List, Literal, Optional

from .encoder import inference as Encoder
from .synthesizer.inference import Synthesizer
from .vocoder.hifigan import inference as gan_vocoder
from .vocoder.wavernn import inference as rnn_vocoder
from .log import logger


def process_text(text: str) -> List[str]:
    punctuation = "！，。、,?!,"  # punctuate and split/clean text
    processed_texts = []
    text = re.sub(r"[{}]+".format(punctuation), "\n", text)
    for processed_text in text.split("\n"):
        if processed_text:
            processed_texts.append(processed_text.strip())
    return processed_texts


class MockingBird:
    @classmethod
    def load_model(
        cls,
        encoder_path: Path,
        gan_vocoder_path: Optional[Path] = None,
        rnn_vocoder_path: Optional[Path] = None,
    ):
        """
        设置 Encoder模型 和 Vocoder模型 路径

        Args:
            encoder_path (Path): Encoder模型路径
            gan_vocoder_path (Path): HifiGan Vocoder模型路径，可选，需要用到 HifiGan 类型时必须填写
            rnn_vocoder_path (Path): WaveRNN Vocoder模型路径，可选，需要用到 WaveRNN 类型时必须填写
        """
        Encoder.load_model(encoder_path)
        if gan_vocoder_path:
            gan_vocoder.load_model(gan_vocoder_path)
        if rnn_vocoder_path:
            rnn_vocoder.load_model(rnn_vocoder_path)
        cls.synthesizer: Optional[Synthesizer] = None

    @classmethod
    def set_synthesizer(cls, synthesizer_path: Path):
        """
        设置Synthesizer模型路径

        Args:
            synthesizer_path (Path): Synthesizer模型路径
        """
        cls.synthesizer = Synthesizer(synthesizer_path)
        logger.info(f"using synthesizer model: {synthesizer_path}")

    @classmethod
    def synthesize(
        cls,
        text: str,
        input_wav: Path,
        vocoder_type: Literal["HifiGan", "WaveRNN"] = "HifiGan",
        style_idx: int = 0,
        min_stop_token: int = 5,
        steps: int = 1000,
    ) -> BytesIO:
        """
        生成语音

        Args:
            text (str): 目标文字
            input_wav (Path): 目标录音路径
            vocoder_type (HifiGan / WaveRNN): Vocoder模型，默认使用HifiGan
            style_idx (int, optional): Style 范围 -1~9，默认为 0
            min_stop_token (int, optional): Accuracy(精度) 范围3~9，默认为 5
            steps (int, optional): MaxLength(最大句长) 范围200~2000，默认为 1000
        """
        if not cls.synthesizer:
            raise Exception("Please set synthesizer path first")

        # Load input wav
        wav, sample_rate = librosa.load(input_wav)

        encoder_wav = Encoder.preprocess_wav(wav, sample_rate)
        embed, _, _ = Encoder.embed_utterance(encoder_wav, return_partials=True)

        # Load input text
        texts = process_text(text)

        # synthesize and vocode
        embeds = [embed] * len(texts)
        specs = cls.synthesizer.synthesize_spectrograms(
            texts,
            embeds,
            style_idx=style_idx,
            min_stop_token=min_stop_token,
            steps=steps,
        )
        spec = np.concatenate(specs, axis=1)
        sample_rate = Synthesizer.sample_rate
        if vocoder_type == "WaveRNN":
            wav, sample_rate = rnn_vocoder.infer_waveform(spec)
        else:
            wav, sample_rate = gan_vocoder.infer_waveform(spec)

        # Return cooked wav
        out = BytesIO()
        wavfile.write(out, sample_rate, wav.astype(np.float32))
        return out
