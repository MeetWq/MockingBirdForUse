import requests
from io import BytesIO
from pathlib import Path


def download(url: str, filename: str):
    from mockingbirdforuse.log import logger

    resp = requests.get(url)
    total_size = int(resp.headers["Content-Length"])
    downloaded_size = 0
    with open(Path(filename), "wb") as f:
        for chunk in resp.iter_content():
            downloaded_size += f.write(chunk)
            logger.trace(
                f"Download progress: {downloaded_size/total_size:.2%} "
                f"({downloaded_size}/{total_size} bytes)"
            )


def test_synthesize():
    from mockingbirdforuse import MockingBird
    from mockingbirdforuse.log import logger

    mockingbird = MockingBird()
    mockingbird.load_model(
        Path("saved_models/encoder.pt"),
        Path("saved_models/hifigan.pt"),
        Path("saved_models/wavernn.pt"),
    )
    logger.info("Downloading synthesizer model")
    download(
        "https://github.com/MeetWq/MockingBirdForUse/releases/download/synthesizer_model/pretrained-11-7-21_75k.pt",
        "synthesizer_model.pt",
    )
    mockingbird.set_synthesizer(Path("synthesizer_model.pt"))
    output = mockingbird.synthesize(
        "欢迎使用拟声鸟工具箱，现已支持中文输入", Path("samples/T0055G0013S0005.wav"), "HifiGan"
    )
    assert isinstance(output, BytesIO)
    output = mockingbird.synthesize(
        "欢迎使用拟声鸟工具箱，现已支持中文输入", Path("samples/T0055G0013S0005.wav"), "WaveRNN"
    )
    assert isinstance(output, BytesIO)
