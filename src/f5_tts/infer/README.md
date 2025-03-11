# 推理

预训练模型检查点可以在 [🤗 Hugging Face](https://huggingface.co/SWivid/F5-TTS) 和 [🤖 Model Scope](https://www.modelscope.cn/models/SWivid/F5-TTS_Emilia-ZH-EN) 上获取，或者在运行推理脚本时自动下载。

**更多由社区贡献的支持多种语言的检查点可以在 [SHARED.md](SHARED.md) 中找到。**

目前支持单次生成**最长30秒**，这是包括提示音频和输出音频在内的**总长度**。不过，你可以向 `infer_cli` 和 `infer_gradio` 提供更长的文本，系统会自动进行分块生成。长参考音频将被**截短到约15秒**。

为避免可能的推理失败，请确保你已经阅读了以下说明：

- 使用时长小于15秒的参考音频，并在末尾留出一些静音（例如1秒）。否则可能会在单词中间截断，导致生成效果不佳。
- 大写字母将被逐字朗读，因此普通单词请使用小写字母。
- 添加一些空格（空白：" "）或标点符号（例如 "," "."）来明确引入停顿。
- 如果希望数字以中文方式朗读，请将数字预处理为中文字符，否则将以英文方式朗读。
- 如果生成输出为空（纯静音），请检查是否安装了ffmpeg（网上有各种教程，博客，视频等）。
- 如果使用早期微调的检查点（仅进行了几次更新），请尝试关闭use_ema。


## Gradio应用

当前支持的功能：

- 基本的TTS与分块推理
- 多风格/多说话人生成
- 由Qwen2.5-3B-Instruct驱动的语音聊天
- [支持更多语言的自定义推理](src/f5_tts/infer/SHARED.md)

命令行指令 `f5-tts_infer-gradio` 等同于 `python src/f5_tts/infer/infer_gradio.py`，它会启动一个Gradio应用（网页界面）进行推理。

该脚本将从Huggingface加载模型检查点。你也可以手动下载文件，并在 `infer_gradio.py` 中更新 `load_model()` 的路径。目前只首先加载TTS模型，如果没有提供 `ref_text` 则会加载ASR模型进行转录，如果使用语音聊天则会加载LLM模型。

更多标志选项：

```bash
# 自动在默认网页浏览器中启动界面
f5-tts_infer-gradio --inbrowser

# 设置应用的根路径，如果它不是从域名的根目录（"/"）提供服务
# 例如，如果应用在 "https://example.com/myapp" 提供服务
f5-tts_infer-gradio --root_path "/myapp"
```

也可以作为更大应用的组件使用：
```python
import gradio as gr
from f5_tts.infer.infer_gradio import app

with gr.Blocks() as main_app:
    gr.Markdown("# 这是在更大的Gradio应用中使用F5-TTS的示例")

    # ... 其他Gradio组件

    app.render()

main_app.launch()
```


## 命令行推理

命令行指令 `f5-tts_infer-cli` 等同于 `python src/f5_tts/infer/infer_cli.py`，这是一个用于推理的命令行工具。

该脚本将从Huggingface加载模型检查点。你也可以手动下载文件，并使用 `--ckpt_file` 指定要加载的模型，或直接在 `infer_cli.py` 中更新。

若要更改vocab.txt，使用 `--vocab_file` 提供你的 `vocab.txt` 文件。

基本上，你可以使用以下标志进行推理：
```bash
# 将 --ref_text 留空 "" 将使用ASR模型进行转录（需要额外的GPU内存）
f5-tts_infer-cli \
--model "F5-TTS" \
--ref_audio "ref_audio.wav" \
--ref_text "参考音频的内容、字幕或转录。" \
--gen_text "你希望TTS模型为你生成的一些文本。"

# 选择声码器
f5-tts_infer-cli --vocoder_name bigvgan --load_vocoder_from_local --ckpt_file <你的检查点路径，例如：ckpts/F5TTS_Base_bigvgan/model_1250000.pt>
f5-tts_infer-cli --vocoder_name vocos --load_vocoder_from_local --ckpt_file <你的检查点路径，例如：ckpts/F5TTS_Base/model_1200000.safetensors>

# 更多说明
f5-tts_infer-cli --help
```

使用 `.toml` 文件可以实现更灵活的用法。

```bash
f5-tts_infer-cli -c custom.toml
```

例如，你可以使用 `.toml` 传递变量，参考 `src/f5_tts/infer/examples/basic/basic.toml`：

```toml
# F5-TTS | E2-TTS
model = "F5-TTS"
ref_audio = "infer/examples/basic/basic_ref_en.wav"
# 如果为空 ""，则自动转录参考音频。
ref_text = "有人称我为自然，也有人称我为大自然母亲。"
gen_text = "我并不在乎你怎么称呼我。我一直是个沉默的观察者，看着物种进化，帝国兴衰。但请记住，我是强大而持久的。"
# 包含要生成文本的文件。忽略上面的文本。
gen_file = ""
remove_silence = false
output_dir = "tests"
```

你也可以利用 `.toml` 文件进行多风格生成，参考 `src/f5_tts/infer/examples/multi/story.toml`。

```toml
# F5-TTS | E2-TTS
model = "F5-TTS"
ref_audio = "infer/examples/multi/main.flac"
# 如果为空 ""，则自动转录参考音频。
ref_text = ""
gen_text = ""
# 包含要生成文本的文件。忽略上面的文本。
gen_file = "infer/examples/multi/story.txt"
remove_silence = true
output_dir = "tests"

[voices.town]
ref_audio = "infer/examples/multi/town.flac"
ref_text = ""

[voices.country]
ref_audio = "infer/examples/multi/country.flac"
ref_text = ""
```
当你想要更换声音时，应该使用 `[main]` `[town]` `[country]` 标记，参考 `src/f5_tts/infer/examples/multi/story.txt`。

## 语音编辑

要测试语音编辑功能，请使用以下命令：

```bash
python src/f5_tts/infer/speech_edit.py
```

## Socket实时客户端

要与socket服务器通信，你需要运行
```bash
python src/f5_tts/socket_server.py
```

<details>
<summary>然后创建客户端进行通信</summary>

```bash
# 如果没有安装PyAudio
sudo apt-get install portaudio19-dev
pip install pyaudio
```

``` python
# 创建socket_client.py
import socket
import asyncio
import pyaudio
import numpy as np
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def listen_to_F5TTS(text, server_ip="localhost", server_port=9998):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    await asyncio.get_event_loop().run_in_executor(None, client_socket.connect, (server_ip, int(server_port)))

    start_time = time.time()
    first_chunk_time = None

    async def play_audio_stream():
        nonlocal first_chunk_time
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paFloat32, channels=1, rate=24000, output=True, frames_per_buffer=2048)

        try:
            while True:
                data = await asyncio.get_event_loop().run_in_executor(None, client_socket.recv, 8192)
                if not data:
                    break
                if data == b"END":
                    logger.info("音频接收结束。")
                    break

                audio_array = np.frombuffer(data, dtype=np.float32)
                stream.write(audio_array.tobytes())

                if first_chunk_time is None:
                    first_chunk_time = time.time()

        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

        logger.info(f"总共用时: {time.time() - start_time:.4f} 秒")

    try:
        data_to_send = f"{text}".encode("utf-8")
        await asyncio.get_event_loop().run_in_executor(None, client_socket.sendall, data_to_send)
        await play_audio_stream()

    except Exception as e:
        logger.error(f"listen_to_F5TTS中出错: {e}")

    finally:
        client_socket.close()


if __name__ == "__main__":
    text_to_send = "作为阅读助手，我熟悉新技术。这些技术对提高训练速度和推理效率方面的性能至关重要。让我们分解一下组件"

    asyncio.run(listen_to_F5TTS(text_to_send))
```

</details>

