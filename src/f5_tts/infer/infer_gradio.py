# ruff: noqa: E402
# 上面允许ruff忽略E402: 模块级导入不在文件顶部

import json
import re
import tempfile
from collections import OrderedDict
from importlib.resources import files

import click
import gradio as gr
import numpy as np
import soundfile as sf
import torchaudio
from cached_path import cached_path
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import spaces

    USING_SPACES = True
except ImportError:
    USING_SPACES = False


def gpu_decorator(func):
    if USING_SPACES:
        return spaces.GPU(func)
    else:
        return func


from f5_tts.model import DiT, UNetT
from f5_tts.infer.utils_infer import (
    load_vocoder,
    load_model,
    preprocess_ref_audio_text,
    infer_process,
    remove_silence_for_generated_wav,
    save_spectrogram,
)


DEFAULT_TTS_MODEL = "F5-TTS"
tts_model_choice = DEFAULT_TTS_MODEL

DEFAULT_TTS_MODEL_CFG = [
    "hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.safetensors",
    "hf://SWivid/F5-TTS/F5TTS_Base/vocab.txt",
    json.dumps(dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)),
]


# 加载模型

vocoder = load_vocoder()


def load_f5tts(ckpt_path=str(cached_path("hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.safetensors"))):
    F5TTS_model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
    return load_model(DiT, F5TTS_model_cfg, ckpt_path)


def load_e2tts(ckpt_path=str(cached_path("hf://SWivid/E2-TTS/E2TTS_Base/model_1200000.safetensors"))):
    E2TTS_model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4)
    return load_model(UNetT, E2TTS_model_cfg, ckpt_path)


def load_custom(ckpt_path: str, vocab_path="", model_cfg=None):
    ckpt_path, vocab_path = ckpt_path.strip(), vocab_path.strip()
    if ckpt_path.startswith("hf://"):
        ckpt_path = str(cached_path(ckpt_path))
    if vocab_path.startswith("hf://"):
        vocab_path = str(cached_path(vocab_path))
    if model_cfg is None:
        model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
    return load_model(DiT, model_cfg, ckpt_path, vocab_file=vocab_path)


F5TTS_ema_model = load_f5tts()
E2TTS_ema_model = load_e2tts() if USING_SPACES else None
custom_ema_model, pre_custom_path = None, ""

chat_model_state = None
chat_tokenizer_state = None


@gpu_decorator
def generate_response(messages, model, tokenizer):
    """使用Qwen生成回复"""
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.95,
    )

    generated_ids = [
        output_ids[len(input_ids) :] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


@gpu_decorator
def infer(
    ref_audio_orig,
    ref_text,
    gen_text,
    model,
    remove_silence,
    cross_fade_duration=0.15,
    nfe_step=32,
    speed=1,
    show_info=gr.Info,
):
    if not ref_audio_orig:
        gr.Warning("请提供参考音频。")
        return gr.update(), gr.update(), ref_text

    if not gen_text.strip():
        gr.Warning("请输入要生成的文本。")
        return gr.update(), gr.update(), ref_text

    ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, ref_text, show_info=show_info)

    if model == "F5-TTS":
        ema_model = F5TTS_ema_model
    elif model == "E2-TTS":
        global E2TTS_ema_model
        if E2TTS_ema_model is None:
            show_info("正在加载E2-TTS模型...")
            E2TTS_ema_model = load_e2tts()
        ema_model = E2TTS_ema_model
    elif isinstance(model, list) and model[0] == "Custom":
        assert not USING_SPACES, "在Spaces中只允许使用官方检查点。"
        global custom_ema_model, pre_custom_path
        if pre_custom_path != model[1]:
            show_info("正在加载自定义TTS模型...")
            custom_ema_model = load_custom(model[1], vocab_path=model[2], model_cfg=model[3])
            pre_custom_path = model[1]
        ema_model = custom_ema_model

    final_wave, final_sample_rate, combined_spectrogram = infer_process(
        ref_audio,
        ref_text,
        gen_text,
        ema_model,
        vocoder,
        cross_fade_duration=cross_fade_duration,
        nfe_step=nfe_step,
        speed=speed,
        show_info=show_info,
        progress=gr.Progress(),
    )

    # 移除静音
    if remove_silence:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            sf.write(f.name, final_wave, final_sample_rate)
            remove_silence_for_generated_wav(f.name)
            final_wave, _ = torchaudio.load(f.name)
        final_wave = final_wave.squeeze().cpu().numpy()

    # 保存频谱图
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_spectrogram:
        spectrogram_path = tmp_spectrogram.name
        save_spectrogram(combined_spectrogram, spectrogram_path)

    return (final_sample_rate, final_wave), spectrogram_path, ref_text


with gr.Blocks() as app_credits:
    gr.Markdown("""
# 致谢

* [mrfakename](https://github.com/fakerybakery) 提供的原始[在线演示](https://huggingface.co/spaces/mrfakename/E2-F5-TTS)
* [RootingInLoad](https://github.com/RootingInLoad) 贡献了初始的分段生成和播客应用探索
* [jpgallegoar](https://github.com/jpgallegoar) 贡献了多种语音类型生成和语音聊天功能
""")
with gr.Blocks() as app_tts:
    gr.Markdown("# 批量语音合成")
    ref_audio_input = gr.Audio(label="参考音频", type="filepath")
    gen_text_input = gr.Textbox(label="待生成文本", lines=10)
    generate_btn = gr.Button("合成语音", variant="primary")
    with gr.Accordion("高级设置", open=False):
        ref_text_input = gr.Textbox(
            label="参考文本",
            info="留空将自动转录参考音频。如果输入文本，将覆盖自动转录结果。",
            lines=2,
        )
        remove_silence = gr.Checkbox(
            label="移除静音",
            info="模型在生成较长音频时倾向于产生静音。需要时可以手动移除静音。请注意，这是一个实验性功能，可能会产生奇怪的结果。这也会增加生成时间。",
            value=False,
        )
        speed_slider = gr.Slider(
            label="语速",
            minimum=0.3,
            maximum=2.0,
            value=1.0,
            step=0.1,
            info="调整音频的播放速度。",
        )
        nfe_slider = gr.Slider(
            label="去噪步数",
            minimum=4,
            maximum=64,
            value=32,
            step=2,
            info="设置去噪步骤的数量。",
        )
        cross_fade_duration_slider = gr.Slider(
            label="交叉淡入淡出时长 (秒)",
            minimum=0.0,
            maximum=1.0,
            value=0.15,
            step=0.01,
            info="设置音频片段之间的交叉淡入淡出时长。",
        )

    audio_output = gr.Audio(label="合成的音频")
    spectrogram_output = gr.Image(label="频谱图")

    @gpu_decorator
    def basic_tts(
        ref_audio_input,
        ref_text_input,
        gen_text_input,
        remove_silence,
        cross_fade_duration_slider,
        nfe_slider,
        speed_slider,
    ):
        audio_out, spectrogram_path, ref_text_out = infer(
            ref_audio_input,
            ref_text_input,
            gen_text_input,
            tts_model_choice,
            remove_silence,
            cross_fade_duration=cross_fade_duration_slider,
            nfe_step=nfe_slider,
            speed=speed_slider,
        )
        return audio_out, spectrogram_path, ref_text_out

    generate_btn.click(
        basic_tts,
        inputs=[
            ref_audio_input,
            ref_text_input,
            gen_text_input,
            remove_silence,
            cross_fade_duration_slider,
            nfe_slider,
            speed_slider,
        ],
        outputs=[audio_output, spectrogram_output, ref_text_input],
    )


def parse_speechtypes_text(gen_text):
    # 查找{speechtype}的模式
    pattern = r"\{(.*?)\}"

    # 按模式分割文本
    tokens = re.split(pattern, gen_text)

    segments = []

    current_style = "常规"

    for i in range(len(tokens)):
        if i % 2 == 0:
            # 这是文本
            text = tokens[i].strip()
            if text:
                segments.append({"style": current_style, "text": text})
        else:
            # 这是风格
            style = tokens[i].strip()
            current_style = style

    return segments


with gr.Blocks() as app_multistyle:
    # 多风格生成部分
    gr.Markdown(
        """
    # 多种语音风格生成

    本部分允许您生成多种语音风格或多个人的声音。按照下面显示的格式输入文本，系统将使用相应的语音风格生成语音。如果未指定，模型将使用常规语音风格。当前语音风格将一直使用，直到指定下一个语音风格。
    """
    )

    with gr.Row():
        gr.Markdown(
            """
            **示例输入1:**                                                                      
            {常规} 你好，我想订一个三明治。                                                         
            {惊讶} 什么？你的面包卖完了？                                                                      
            {悲伤} 我真的很想要一个三明治...                                                              
            {愤怒} 你知道吗，去你的小店！                                                                       
            {低语} 我现在要回家哭了。                                                                           
            {大喊} 为什么是我啊？!                                                                         
            """
        )

        gr.Markdown(
            """
            **示例输入2:**                                                                                
            {说话人1_开心} 你好，我想订一个三明治。                                                            
            {说话人2_常规} 抱歉，我们的面包卖完了。                                                                                
            {说话人1_悲伤} 我真的很想要一个三明治...                                                                             
            {说话人2_低语} 我给你最后一个我藏起来的。                                                                     
            """
        )

    gr.Markdown(
        "为每种语音风格上传不同的音频片段。第一种语音风格是必需的。您可以通过点击'添加语音风格'按钮添加额外的语音风格。"
    )

    # 常规语音风格（必需）
    with gr.Row() as regular_row:
        with gr.Column():
            regular_name = gr.Textbox(value="常规", label="语音风格名称")
            regular_insert = gr.Button("插入标签", variant="secondary")
        regular_audio = gr.Audio(label="常规参考音频", type="filepath")
        regular_ref_text = gr.Textbox(label="参考文本（常规）", lines=2)

    # 常规语音风格（最多100个）
    max_speech_types = 100
    speech_type_rows = [regular_row]
    speech_type_names = [regular_name]
    speech_type_audios = [regular_audio]
    speech_type_ref_texts = [regular_ref_text]
    speech_type_delete_btns = [None]
    speech_type_insert_btns = [regular_insert]

    # 额外的语音风格（还有99个）
    for i in range(max_speech_types - 1):
        with gr.Row(visible=False) as row:
            with gr.Column():
                name_input = gr.Textbox(label="语音风格名称")
                delete_btn = gr.Button("删除风格", variant="secondary")
                insert_btn = gr.Button("插入标签", variant="secondary")
            audio_input = gr.Audio(label="参考音频", type="filepath")
            ref_text_input = gr.Textbox(label="参考文本", lines=2)
        speech_type_rows.append(row)
        speech_type_names.append(name_input)
        speech_type_audios.append(audio_input)
        speech_type_ref_texts.append(ref_text_input)
        speech_type_delete_btns.append(delete_btn)
        speech_type_insert_btns.append(insert_btn)

    # 添加语音风格的按钮
    add_speech_type_btn = gr.Button("添加语音风格")

    # 跟踪语音风格的自动递增，不回滚
    speech_type_count = 1

    # 添加语音风格的函数
    def add_speech_type_fn():
        row_updates = [gr.update() for _ in range(max_speech_types)]
        global speech_type_count
        if speech_type_count < max_speech_types:
            row_updates[speech_type_count] = gr.update(visible=True)
            speech_type_count += 1
        else:
            gr.Warning("已达到最大语音风格数量。请考虑重启应用。")
        return row_updates

    add_speech_type_btn.click(add_speech_type_fn, outputs=speech_type_rows)

    # 删除语音风格的函数
    def delete_speech_type_fn():
        return gr.update(visible=False), None, None, None

    # 更新删除按钮点击
    for i in range(1, len(speech_type_delete_btns)):
        speech_type_delete_btns[i].click(
            delete_speech_type_fn,
            outputs=[speech_type_rows[i], speech_type_names[i], speech_type_audios[i], speech_type_ref_texts[i]],
        )

    # 文本输入提示
    gen_text_input_multistyle = gr.Textbox(
        label="待生成文本",
        lines=10,
        placeholder="输入带有说话人名称（或情感类型）的脚本，例如：\n\n{常规} 你好，我想订一个三明治。\n{惊讶} 什么？你的面包卖完了？\n{悲伤} 我真的很想要一个三明治...\n{愤怒} 你知道吗，去你的小店！\n{低语} 我现在要回家哭了。\n{大喊} 为什么是我啊？!",
    )

    def make_insert_speech_type_fn(index):
        def insert_speech_type_fn(current_text, speech_type_name):
            current_text = current_text or ""
            speech_type_name = speech_type_name or "无"
            updated_text = current_text + f"{{{speech_type_name}}} "
            return updated_text

        return insert_speech_type_fn

    for i, insert_btn in enumerate(speech_type_insert_btns):
        insert_fn = make_insert_speech_type_fn(i)
        insert_btn.click(
            insert_fn,
            inputs=[gen_text_input_multistyle, speech_type_names[i]],
            outputs=gen_text_input_multistyle,
        )

    with gr.Accordion("高级设置", open=False):
        remove_silence_multistyle = gr.Checkbox(
            label="移除静音",
            value=True,
        )

    # 生成按钮
    generate_multistyle_btn = gr.Button("生成多风格语音", variant="primary")

    # 输出音频
    audio_output_multistyle = gr.Audio(label="合成的音频")

    @gpu_decorator
    def generate_multistyle_speech(
        gen_text,
        *args,
    ):
        speech_type_names_list = args[:max_speech_types]
        speech_type_audios_list = args[max_speech_types : 2 * max_speech_types]
        speech_type_ref_texts_list = args[2 * max_speech_types : 3 * max_speech_types]
        remove_silence = args[3 * max_speech_types]
        # 收集语音风格及其音频到字典中
        speech_types = OrderedDict()

        ref_text_idx = 0
        for name_input, audio_input, ref_text_input in zip(
            speech_type_names_list, speech_type_audios_list, speech_type_ref_texts_list
        ):
            if name_input and audio_input:
                speech_types[name_input] = {"audio": audio_input, "ref_text": ref_text_input}
            else:
                speech_types[f"@{ref_text_idx}@"] = {"audio": "", "ref_text": ""}
            ref_text_idx += 1

        # 将gen_text解析为段落
        segments = parse_speechtypes_text(gen_text)

        # 为每个段落生成语音
        generated_audio_segments = []
        current_style = "常规"

        for segment in segments:
            style = segment["style"]
            text = segment["text"]

            if style in speech_types:
                current_style = style
            else:
                gr.Warning(f"风格 {style} 不可用，将使用常规风格作为默认值。")
                current_style = "常规"

            try:
                ref_audio = speech_types[current_style]["audio"]
            except KeyError:
                gr.Warning(f"请为风格 {current_style} 提供参考音频。")
                return [None] + [speech_types[style]["ref_text"] for style in speech_types]
            ref_text = speech_types[current_style].get("ref_text", "")

            # 为这个段落生成语音
            audio_out, _, ref_text_out = infer(
                ref_audio, ref_text, text, tts_model_choice, remove_silence, 0, show_info=print
            )  # show_info=print 生成时不弹出顶部提示
            sr, audio_data = audio_out

            generated_audio_segments.append(audio_data)
            speech_types[current_style]["ref_text"] = ref_text_out

        # 连接所有音频段落
        if generated_audio_segments:
            final_audio_data = np.concatenate(generated_audio_segments)
            return [(sr, final_audio_data)] + [speech_types[style]["ref_text"] for style in speech_types]
        else:
            gr.Warning("未生成任何音频。")
            return [None] + [speech_types[style]["ref_text"] for style in speech_types]

    generate_multistyle_btn.click(
        generate_multistyle_speech,
        inputs=[
            gen_text_input_multistyle,
        ]
        + speech_type_names
        + speech_type_audios
        + speech_type_ref_texts
        + [
            remove_silence_multistyle,
        ],
        outputs=[audio_output_multistyle] + speech_type_ref_texts,
    )

    # 验证函数，如果缺少语音风格则禁用生成按钮
    def validate_speech_types(gen_text, regular_name, *args):
        speech_type_names_list = args

        # 收集可用的语音风格名称
        speech_types_available = set()
        if regular_name:
            speech_types_available.add(regular_name)
        for name_input in speech_type_names_list:
            if name_input:
                speech_types_available.add(name_input)

        # 解析gen_text获取使用的语音风格
        segments = parse_speechtypes_text(gen_text)
        speech_types_in_text = set(segment["style"] for segment in segments)

        # 检查文本中的所有语音风格是否可用
        missing_speech_types = speech_types_in_text - speech_types_available

        if missing_speech_types:
            # 禁用生成按钮
            return gr.update(interactive=False)
        else:
            # 启用生成按钮
            return gr.update(interactive=True)

    gen_text_input_multistyle.change(
        validate_speech_types,
        inputs=[gen_text_input_multistyle, regular_name] + speech_type_names,
        outputs=generate_multistyle_btn,
    )


with gr.Blocks() as app_chat:
    gr.Markdown(
        """
# 语音聊天
使用您的参考声音与AI进行对话！
1. 上传参考音频片段及其可选的转录文本。
2. 加载聊天模型。
3. 通过麦克风录制您的消息。
4. AI将使用参考声音回复。
"""
    )

    if not USING_SPACES:
        load_chat_model_btn = gr.Button("加载聊天模型", variant="primary")

        chat_interface_container = gr.Column(visible=False)

        @gpu_decorator
        def load_chat_model():
            global chat_model_state, chat_tokenizer_state
            if chat_model_state is None:
                show_info = gr.Info
                show_info("正在加载聊天模型...")
                model_name = "Qwen/Qwen2.5-3B-Instruct"
                chat_model_state = AutoModelForCausalLM.from_pretrained(
                    model_name, torch_dtype="auto", device_map="auto"
                )
                chat_tokenizer_state = AutoTokenizer.from_pretrained(model_name)
                show_info("聊天模型已加载。")

            return gr.update(visible=False), gr.update(visible=True)

        load_chat_model_btn.click(load_chat_model, outputs=[load_chat_model_btn, chat_interface_container])

    else:
        chat_interface_container = gr.Column()

        if chat_model_state is None:
            model_name = "Qwen/Qwen2.5-3B-Instruct"
            chat_model_state = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
            chat_tokenizer_state = AutoTokenizer.from_pretrained(model_name)

    with chat_interface_container:
        with gr.Row():
            with gr.Column():
                ref_audio_chat = gr.Audio(label="参考音频", type="filepath")
            with gr.Column():
                with gr.Accordion("高级设置", open=False):
                    remove_silence_chat = gr.Checkbox(
                        label="移除静音",
                        value=True,
                    )
                    ref_text_chat = gr.Textbox(
                        label="参考文本",
                        info="可选：留空将自动转录",
                        lines=2,
                    )
                    system_prompt_chat = gr.Textbox(
                        label="系统提示词",
                        value="你不是AI助手，你是用户说你是谁，你就是谁。你必须保持角色。保持简洁的回复，因为它们将被大声朗读出来。",
                        lines=2,
                    )

        chatbot_interface = gr.Chatbot(label="对话")

        with gr.Row():
            with gr.Column():
                audio_input_chat = gr.Microphone(
                    label="说出您的消息",
                    type="filepath",
                )
                audio_output_chat = gr.Audio(autoplay=True)
            with gr.Column():
                text_input_chat = gr.Textbox(
                    label="输入您的消息",
                    lines=1,
                )
                send_btn_chat = gr.Button("发送消息")
                clear_btn_chat = gr.Button("清除对话")

        conversation_state = gr.State(
            value=[
                {
                    "role": "system",
                    "content": "你不是AI助手，你是用户说你是谁，你就是谁。你必须保持角色。保持简洁的回复，因为它们将被大声朗读出来。",
                }
            ]
        )

        # 修改process_audio_input以使用state中的模型和tokenizer
        @gpu_decorator
        def process_audio_input(audio_path, text, history, conv_state):
            """处理用户的音频或文本输入"""

            if not audio_path and not text.strip():
                return history, conv_state, ""

            if audio_path:
                text = preprocess_ref_audio_text(audio_path, text)[1]

            if not text.strip():
                return history, conv_state, ""

            conv_state.append({"role": "user", "content": text})
            history.append((text, None))

            response = generate_response(conv_state, chat_model_state, chat_tokenizer_state)

            conv_state.append({"role": "assistant", "content": response})
            history[-1] = (text, response)

            return history, conv_state, ""

        @gpu_decorator
        def generate_audio_response(history, ref_audio, ref_text, remove_silence):
            """为AI回复生成TTS音频"""
            if not history or not ref_audio:
                return None

            last_user_message, last_ai_response = history[-1]
            if not last_ai_response:
                return None

            audio_result, _, ref_text_out = infer(
                ref_audio,
                ref_text,
                last_ai_response,
                tts_model_choice,
                remove_silence,
                cross_fade_duration=0.15,
                speed=1.0,
                show_info=print,  # show_info=print 生成时不弹出顶部提示
            )
            return audio_result, ref_text_out

        def clear_conversation():
            """重置对话"""
            return [], [
                {
                    "role": "system",
                    "content": "你不是AI助手，你是用户说你是谁，你就是谁。你必须保持角色。保持简洁的回复，因为它们将被大声朗读出来。",
                }
            ]

        def update_system_prompt(new_prompt):
            """更新系统提示词并重置对话"""
            new_conv_state = [{"role": "system", "content": new_prompt}]
            return [], new_conv_state

        # 处理音频输入
        audio_input_chat.stop_recording(
            process_audio_input,
            inputs=[audio_input_chat, text_input_chat, chatbot_interface, conversation_state],
            outputs=[chatbot_interface, conversation_state],
        ).then(
            generate_audio_response,
            inputs=[chatbot_interface, ref_audio_chat, ref_text_chat, remove_silence_chat],
            outputs=[audio_output_chat, ref_text_chat],
        ).then(
            lambda: None,
            None,
            audio_input_chat,
        )

        # 处理文本输入
        text_input_chat.submit(
            process_audio_input,
            inputs=[audio_input_chat, text_input_chat, chatbot_interface, conversation_state],
            outputs=[chatbot_interface, conversation_state],
        ).then(
            generate_audio_response,
            inputs=[chatbot_interface, ref_audio_chat, ref_text_chat, remove_silence_chat],
            outputs=[audio_output_chat, ref_text_chat],
        ).then(
            lambda: None,
            None,
            text_input_chat,
        )

        # 处理发送按钮
        send_btn_chat.click(
            process_audio_input,
            inputs=[audio_input_chat, text_input_chat, chatbot_interface, conversation_state],
            outputs=[chatbot_interface, conversation_state],
        ).then(
            generate_audio_response,
            inputs=[chatbot_interface, ref_audio_chat, ref_text_chat, remove_silence_chat],
            outputs=[audio_output_chat, ref_text_chat],
        ).then(
            lambda: None,
            None,
            text_input_chat,
        )

        # 处理清除按钮
        clear_btn_chat.click(
            clear_conversation,
            outputs=[chatbot_interface, conversation_state],
        )

        # 处理系统提示词变更并重置对话
        system_prompt_chat.change(
            update_system_prompt,
            inputs=system_prompt_chat,
            outputs=[chatbot_interface, conversation_state],
        )


with gr.Blocks() as app:
    gr.Markdown(
        f"""
# E2/F5 语音合成

这是{"F5 TTS的本地Web界面" if not USING_SPACES else "F5-TTS的在线演示"}，支持高级批处理功能。本应用支持以下TTS模型：

* [F5-TTS](https://arxiv.org/abs/2410.06885)（使用流匹配技术实现流畅且忠实的语音合成）
* [E2 TTS](https://arxiv.org/abs/2406.18009)（简单易用的完全非自回归零样本TTS）

当前的检查点支持英语和中文。

如果遇到问题，请尝试将参考音频转换为WAV或MP3格式，使用右下角的✂️工具将其剪辑到15秒（否则可能会得到非最优的自动裁剪结果）。

**注意：如果未提供参考文本，将使用Whisper自动转录。为获得最佳效果，请保持参考片段简短（<15秒）。确保在生成之前音频已完全上传。**
"""
    )

    last_used_custom = files("f5_tts").joinpath("infer/.cache/last_used_custom_model_info.txt")

    def load_last_used_custom():
        try:
            custom = []
            with open(last_used_custom, "r", encoding="utf-8") as f:
                for line in f:
                    custom.append(line.strip())
            return custom
        except FileNotFoundError:
            last_used_custom.parent.mkdir(parents=True, exist_ok=True)
            return DEFAULT_TTS_MODEL_CFG

    def switch_tts_model(new_choice):
        global tts_model_choice
        if new_choice == "自定义":  # 网页刷新时覆盖
            custom_ckpt_path, custom_vocab_path, custom_model_cfg = load_last_used_custom()
            tts_model_choice = ["Custom", custom_ckpt_path, custom_vocab_path, json.loads(custom_model_cfg)]
            return (
                gr.update(visible=True, value=custom_ckpt_path),
                gr.update(visible=True, value=custom_vocab_path),
                gr.update(visible=True, value=custom_model_cfg),
            )
        else:
            tts_model_choice = new_choice
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

    def set_custom_model(custom_ckpt_path, custom_vocab_path, custom_model_cfg):
        global tts_model_choice
        tts_model_choice = ["Custom", custom_ckpt_path, custom_vocab_path, json.loads(custom_model_cfg)]
        with open(last_used_custom, "w", encoding="utf-8") as f:
            f.write(custom_ckpt_path + "\n" + custom_vocab_path + "\n" + custom_model_cfg + "\n")

    with gr.Row():
        if not USING_SPACES:
            choose_tts_model = gr.Radio(
                choices=[DEFAULT_TTS_MODEL, "E2-TTS", "自定义"], label="选择TTS模型", value=DEFAULT_TTS_MODEL
            )
        else:
            choose_tts_model = gr.Radio(
                choices=[DEFAULT_TTS_MODEL, "E2-TTS"], label="选择TTS模型", value=DEFAULT_TTS_MODEL
            )
        custom_ckpt_path = gr.Dropdown(
            choices=[DEFAULT_TTS_MODEL_CFG[0]],
            value=load_last_used_custom()[0],
            allow_custom_value=True,
            label="模型：本地路径 | hf://用户id/仓库id/模型检查点",
            visible=False,
        )
        custom_vocab_path = gr.Dropdown(
            choices=[DEFAULT_TTS_MODEL_CFG[1]],
            value=load_last_used_custom()[1],
            allow_custom_value=True,
            label="词汇表：本地路径 | hf://用户id/仓库id/词汇文件",
            visible=False,
        )
        custom_model_cfg = gr.Dropdown(
            choices=[
                DEFAULT_TTS_MODEL_CFG[2],
                json.dumps(dict(dim=768, depth=18, heads=12, ff_mult=2, text_dim=512, conv_layers=4)),
            ],
            value=load_last_used_custom()[2],
            allow_custom_value=True,
            label="配置：以字典形式",
            visible=False,
        )
    choose_tts_model.change(
        switch_tts_model,
        inputs=[choose_tts_model],
        outputs=[custom_ckpt_path, custom_vocab_path, custom_model_cfg],
        show_progress="hidden",
    )
    custom_ckpt_path.change(
        set_custom_model,
        inputs=[custom_ckpt_path, custom_vocab_path, custom_model_cfg],
        show_progress="hidden",
    )
    custom_vocab_path.change(
        set_custom_model,
        inputs=[custom_ckpt_path, custom_vocab_path, custom_model_cfg],
        show_progress="hidden",
    )
    custom_model_cfg.change(
        set_custom_model,
        inputs=[custom_ckpt_path, custom_vocab_path, custom_model_cfg],
        show_progress="hidden",
    )

    gr.TabbedInterface(
        [app_tts, app_multistyle, app_chat, app_credits],
        ["基础TTS", "多风格语音", "语音聊天", "致谢"],
    )


@click.command()
@click.option("--port", "-p", default=None, type=int, help="运行应用的端口")
@click.option("--host", "-H", default=None, help="运行应用的主机")
@click.option(
    "--share",
    "-s",
    default=False,
    is_flag=True,
    help="通过Gradio共享链接分享应用",
)
@click.option("--api", "-a", default=True, is_flag=True, help="允许API访问")
@click.option(
    "--root_path",
    "-r",
    default=None,
    type=str,
    help='应用的根路径（或"挂载点"），如果它不是从域的根目录（"/"）提供服务。通常在应用程序位于将请求转发到应用程序的反向代理后面时使用，例如为在"https://example.com/myapp"提供服务的应用程序设置"/myapp"或完整URL。',
)
@click.option(
    "--inbrowser",
    "-i",
    is_flag=True,
    default=False,
    help="自动在默认网络浏览器中启动界面",
)
def main(port, host, share, api, root_path, inbrowser):
    global app
    print("正在启动应用...")
    app.queue(api_open=api).launch(
        server_name=host,
        server_port=port,
        share=share,
        show_api=api,
        root_path=root_path,
        inbrowser=inbrowser,
    )


if __name__ == "__main__":
    if not USING_SPACES:
        main()
    else:
        app.queue().launch()