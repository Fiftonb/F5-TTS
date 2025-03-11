# F5-TTS: 一个能流畅、忠实且自然地"讲故事"的语音合成系统

[![python](https://img.shields.io/badge/Python-3.10-brightgreen)](https://github.com/SWivid/F5-TTS)
[![arXiv](https://img.shields.io/badge/arXiv-2410.06885-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2410.06885)
[![demo](https://img.shields.io/badge/GitHub-Demo%20page-orange.svg)](https://swivid.github.io/F5-TTS/)
[![hfspace](https://img.shields.io/badge/🤗-Space%20demo-yellow)](https://huggingface.co/spaces/mrfakename/E2-F5-TTS)
[![msspace](https://img.shields.io/badge/🤖-Space%20demo-blue)](https://modelscope.cn/studios/modelscope/E2-F5-TTS)
[![lab](https://img.shields.io/badge/X--LANCE-Lab-grey?labelColor=lightgrey)](https://x-lance.sjtu.edu.cn/)
[![lab](https://img.shields.io/badge/Peng%20Cheng-Lab-grey?labelColor=lightgrey)](https://www.pcl.ac.cn)
<!-- <img src="https://github.com/user-attachments/assets/12d7749c-071a-427c-81bf-b87b91def670" alt="Watermark" style="width: 40px; height: auto"> -->

**F5-TTS**: 采用ConvNeXt V2的扩散变换器，训练和推理更快。

**E2 TTS**: Flat-UNet变换器，最接近[论文](https://arxiv.org/abs/2406.18009)的复现版本。

**Sway Sampling**: 推理时的流采样策略，极大提升性能。

### 感谢所有贡献者！

## 新闻
- **2024/10/08**: F5-TTS与E2 TTS基础模型已上线[🤗 Hugging Face](https://huggingface.co/SWivid/F5-TTS)、[🤖 模型库](https://www.modelscope.cn/models/SWivid/F5-TTS_Emilia-ZH-EN)和[🟣 智谱](https://wisemodel.cn/models/SJTU_X-LANCE/F5-TTS_Emilia-ZH-EN)。

## 安装

### 如需创建独立环境

```bash
# 创建Python 3.10的conda环境（也可以使用virtualenv）
conda create -n f5-tts python=3.10
conda activate f5-tts
```

### 根据您的设备安装PyTorch

<details>
<summary>NVIDIA GPU</summary>

> ```bash
> # 根据您的CUDA版本安装PyTorch，例如
> pip install torch==2.3.0+cu118 torchaudio==2.3.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
> ```

</details>

<details>
<summary>AMD GPU</summary>

> ```bash
> # 根据您的ROCm版本安装PyTorch（仅限Linux），例如
> pip install torch==2.5.1+rocm6.2 torchaudio==2.5.1+rocm6.2 --extra-index-url https://download.pytorch.org/whl/rocm6.2
> ```

</details>

<details>
<summary>Intel GPU</summary>

> ```bash
> # 根据您的XPU版本安装PyTorch，例如
> # 必须先安装Intel® Deep Learning Essentials或Intel® oneAPI Base Toolkit
> pip install torch torchaudio --index-url https://download.pytorch.org/whl/test/xpu
> 
> # Intel GPU也可通过IPEX（Intel® Extension for PyTorch）支持
> # IPEX不需要安装Intel® Deep Learning Essentials或Intel® oneAPI Base Toolkit
> # 参见：https://pytorch-extension.intel.com/installation?request=platform
> ```

</details>

<details>
<summary>Apple Silicon</summary>

> ```bash
> # 安装稳定版PyTorch，例如
> pip install torch torchaudio
> ```

</details>

### 然后您可以从以下选项中选择一种：

> ### 1. 作为pip包安装（如果只用于推理）
> 
> ```bash
> pip install git+https://github.com/SWivid/F5-TTS.git
> ```
> 
> ### 2. 本地可编辑安装（如果还需要训练、微调）
> 
> ```bash
> git clone https://github.com/SWivid/F5-TTS.git
> cd F5-TTS
> # git submodule update --init --recursive  # （可选，如需bigvgan）
> pip install -e .
> ```

### 也可使用Docker
```bash
# 从Dockerfile构建
docker build -t f5tts:v1 .

# 或从GitHub容器注册表拉取
docker pull ghcr.io/swivid/f5-tts:main
```


## 推理

### 1. Gradio应用

当前支持的功能：

- 基础TTS与分块推理
- 多风格/多说话人生成
- 由Qwen2.5-3B-Instruct驱动的语音聊天
- [支持更多语言的自定义推理](src/f5_tts/infer/SHARED.md)

```bash
# 启动Gradio应用（网页界面）
f5-tts_infer-gradio

# 指定端口/主机
f5-tts_infer-gradio --port 7860 --host 0.0.0.0

# 生成分享链接
f5-tts_infer-gradio --share
```

<details>
<summary>NVIDIA设备的docker-compose文件示例</summary>

```yaml
services:
  f5-tts:
    image: ghcr.io/swivid/f5-tts:main
    ports:
      - "7860:7860"
    environment:
      GRADIO_SERVER_PORT: 7860
    entrypoint: ["f5-tts_infer-gradio", "--port", "7860", "--host", "0.0.0.0"]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  f5-tts:
    driver: local
```

</details>

### 2. 命令行推理

```bash
# 使用参数运行
# 将--ref_text设为""会让ASR模型进行转录（需要额外的GPU内存）
f5-tts_infer-cli \
--model "F5-TTS" \
--ref_audio "ref_audio.wav" \
--ref_text "参考音频的内容、字幕或转录。" \
--gen_text "您希望TTS模型为您生成的文本。"

# 使用默认设置运行。src/f5_tts/infer/examples/basic/basic.toml
f5-tts_infer-cli
# 或使用您自己的.toml文件
f5-tts_infer-cli -c custom.toml

# 多声音。参见src/f5_tts/infer/README.md
f5-tts_infer-cli -c src/f5_tts/infer/examples/multi/story.toml
```

### 3. 更多说明

- 为了获得更好的生成结果，请花点时间阅读[详细指南](src/f5_tts/infer)。
- [Issues](https://github.com/SWivid/F5-TTS/issues?q=is%3Aissue)非常有用，请尝试通过正确搜索遇到问题的关键词来找到解决方案。如果没有找到答案，请随时提出问题。


## 训练

### 1. Gradio应用

阅读[训练与微调指南](src/f5_tts/train)获取更多说明。

```bash
# 使用Gradio网页界面快速开始
f5-tts_finetune-gradio
```


## [评估](src/f5_tts/eval)


## 开发

使用pre-commit确保代码质量（将自动运行代码检查器和格式化工具）

```bash
pip install pre-commit
pre-commit install
```

提交拉取请求时，每次提交前，运行：

```bash
pre-commit run --all-files
```

注意：某些模型组件有E722的检查例外，以适应张量表示法


## 致谢

- [E2-TTS](https://arxiv.org/abs/2406.18009)出色的工作，简单且有效
- [Emilia](https://arxiv.org/abs/2407.05361)、[WenetSpeech4TTS](https://arxiv.org/abs/2406.05763)、[LibriTTS](https://arxiv.org/abs/1904.02882)、[LJSpeech](https://keithito.com/LJ-Speech-Dataset/)宝贵的数据集
- [lucidrains](https://github.com/lucidrains)的初始CFM结构以及与[bfs18](https://github.com/bfs18)的讨论
- [SD3](https://arxiv.org/abs/2403.03206)和[Hugging Face diffusers](https://github.com/huggingface/diffusers)的DiT和MMDiT代码结构
- [torchdiffeq](https://github.com/rtqichen/torchdiffeq)作为ODE求解器，[Vocos](https://huggingface.co/charactr/vocos-mel-24khz)和[BigVGAN](https://github.com/NVIDIA/BigVGAN)作为声码器
- [FunASR](https://github.com/modelscope/FunASR)、[faster-whisper](https://github.com/SYSTRAN/faster-whisper)、[UniSpeech](https://github.com/microsoft/UniSpeech)、[SpeechMOS](https://github.com/tarepan/SpeechMOS)评估工具
- [ctc-forced-aligner](https://github.com/MahmoudAshraf97/ctc-forced-aligner)用于语音编辑测试
- [mrfakename](https://x.com/realmrfakename)的huggingface space演示
- [f5-tts-mlx](https://github.com/lucasnewman/f5-tts-mlx/tree/main) [Lucas Newman](https://github.com/lucasnewman)使用MLX框架的实现
- [F5-TTS-ONNX](https://github.com/DakeQQ/F5-TTS-ONNX) [DakeQQ](https://github.com/DakeQQ)的ONNX Runtime版本

## 引用
如果我们的工作和代码对您有用，请引用：
```
@article{chen-etal-2024-f5tts,
      title={F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching}, 
      author={Yushen Chen and Zhikang Niu and Ziyang Ma and Keqi Deng and Chunhui Wang and Jian Zhao and Kai Yu and Xie Chen},
      journal={arXiv preprint arXiv:2410.06885},
      year={2024},
}
```
## 许可证

我们的代码基于MIT许可证发布。预训练模型基于CC-BY-NC许可证，这是由于训练数据Emilia是一个野外数据集。对于由此可能造成的不便，我们深表歉意。
