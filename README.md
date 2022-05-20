## 实时语音克隆 - 中文/普通话
![mockingbird](https://user-images.githubusercontent.com/12797292/131216767-6eb251d6-14fc-4951-8324-2722f0cd4c63.jpg)

原项目地址：[MockingBird](https://github.com/babysor/MockingBird)


### 使用示例

```python
from pathlib import Path
from mockingbirdforuse import MockingBird

MockingBird.load_model(
    Path("saved_models/encoder.pt"),
    Path("saved_models/hifigan.pt"),
    Path("saved_models/wavernn.pt"),
)
MockingBird.set_synthesizer(Path("azusa.pt"))
output = MockingBird.synthesize("主播不是你的电梓播放器", Path("record.wav"))  # output type: BytesIO
```
