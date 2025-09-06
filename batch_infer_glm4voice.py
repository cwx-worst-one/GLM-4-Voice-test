#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import uuid
import argparse
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm
from transformers import AutoTokenizer, WhisperFeatureExtractor, AutoModel

# 你项目里的依赖
sys.path.insert(0, "./cosyvoice")
sys.path.insert(0, "./third_party/Matcha-TTS")
from speech_tokenizer.modeling_whisper import WhisperVQEncoder
from speech_tokenizer.utils import extract_speech_token
from flow_inference import AudioDecoder

SAVE_SR = 22050  # 输出采样率


# ----------------------------
# Model loading / utilities
# ----------------------------
def load_models(args):
    glm_tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    glm_model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True).eval().to(args.device)

    whisper_model = WhisperVQEncoder.from_pretrained(args.tokenizer_path).eval().to(args.device)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.tokenizer_path)

    audio_decoder = AudioDecoder(
        config_path=os.path.join(args.flow_path, "config.yaml"),
        flow_ckpt_path=os.path.join(args.flow_path, "flow.pt"),
        hift_ckpt_path=os.path.join(args.flow_path, "hift.pt"),
        device=args.device,
    )

    # 预计算标记，避免循环中重复做
    audio_offset = glm_tokenizer.convert_tokens_to_ids("<|audio_0|>")
    end_of_question_tokens = glm_tokenizer.encode("streaming_transcription\n")[-5:]

    return {
        "glm_tokenizer": glm_tokenizer,
        "glm_model": glm_model,
        "whisper_model": whisper_model,
        "feature_extractor": feature_extractor,
        "audio_decoder": audio_decoder,
        "audio_offset": audio_offset,
        "end_q": end_of_question_tokens,
    }


def process_audio_to_tokens(audio_path, whisper_model, feature_extractor):
    tokens = extract_speech_token(whisper_model, feature_extractor, [audio_path])[0]
    if len(tokens) == 0:
        raise ValueError(f"No audio tokens extracted for: {audio_path}")
    audio_tokens = "".join([f"<|audio_{x}|>" for x in tokens])
    return "<|begin_of_audio|>" + audio_tokens + "<|end_of_audio|>"


@torch.no_grad()
def generate_ids(prompt, tokenizer, model, device, temperature, top_p, max_new_tokens):
    inputs = tokenizer([prompt], return_tensors="pt").to(device)
    return model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )


def parse_generation(output_ids, start_marker_ids, audio_offset, tokenizer):
    # 找到 "<|assistant|>streaming_transcription\n" 之后的起始位置
    ids = output_ids[0].tolist()
    # 有时 tokenizer.encode 结果长度可能变化，稳妥起见搜索切片
    start_index = None
    m = len(start_marker_ids)
    for i in range(len(ids) - m + 1):
        if ids[i : i + m] == start_marker_ids:
            start_index = i + m
            break
    if start_index is None:
        start_index = 0  # 兜底

    text_ids, audio_ids = [], []
    for token_id in output_ids[0][start_index:]:
        tid = int(token_id)
        if tid >= audio_offset:
            audio_ids.append(tid - audio_offset)
        else:
            text_ids.append(tid)

    text = tokenizer.decode(text_ids, spaces_between_special_tokens=False)
    # 清掉可能的后续 user 段
    text = text.split("<|user|>")[0].strip()
    return text, audio_ids


def vocode(audio_ids, audio_decoder, device):
    if len(audio_ids) == 0:
        return None, None
    tts_token = torch.tensor(audio_ids, device=device).unsqueeze(0)
    wav, _ = audio_decoder.token2wav(tts_token, uuid=str(uuid.uuid4()), finalize=True)
    # 返回 CPU tensor, sr
    return wav.squeeze().cpu()


def save_one_result(utt, text, wav, sr, out_text_fh, audio_out_dir):
    # 写文本 jsonl
    out_text_fh.write(json.dumps({"utt": utt, "output": text}, ensure_ascii=False) + "\n")
    out_text_fh.flush()

    # 写音频到 {utt}_answer.wav
    audio_out_dir.mkdir(parents=True, exist_ok=True)
    wav_name = f"{utt}_answer.wav"
    torchaudio.save(str(audio_out_dir / wav_name), wav.unsqueeze(0), sr, format="wav")


# ----------------------------
# Batch runners
# ----------------------------
def run_text_mode(args, models):
    in_path = Path(args.input_path)
    assert in_path.is_file(), f"JSONL not found: {in_path}"

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    text_out = outdir / "text_output.jsonl"
    audio_out_dir = outdir / "audio_output"

    with in_path.open("r", encoding="utf-8") as fin:
        lines = fin.readlines()

    pbar = tqdm(total=len(lines), desc="Batch(text) infer", ncols=100)
    with open(text_out, "w", encoding="utf-8") as fout:
        for i, line in enumerate(lines):
            try:
                obj = json.loads(line.strip())
            except Exception:
                pbar.update(1)
                continue

            # 取字段
            user_text = str(obj.get(args.field, "")).strip()
            if not user_text:
                pbar.update(1)
                continue

            utt = obj.get("utt", f"sample_{i}")  # 取 utt 字段，没有则用 sample_i

            # 组 prompt
            system_prompt = (
                "User will provide you with a text instruction. Do it step by step. "
                "First, think about the instruction and respond in a interleaved manner, "
                "with 13 text token followed by 26 audio tokens."
            )
            full_prompt = f"<|system|>\n{system_prompt}<|user|>\n{user_text}<|assistant|>streaming_transcription\n"

            # 生成
            out_ids = generate_ids(
                full_prompt,
                models["glm_tokenizer"],
                models["glm_model"],
                args.device,
                args.temperature,
                args.top_p,
                args.max_new_token,
            )

            # 解析
            text, audio_ids = parse_generation(out_ids, models["end_q"], models["audio_offset"], models["glm_tokenizer"])

            # 声码
            wav = vocode(audio_ids, models["audio_decoder"], args.device)
            if wav is None:
                # 即便声码失败也把文本写出来
                fout.write(json.dumps({"utt": utt, "output": text}, ensure_ascii=False) + "\n")
                fout.flush()
            else:
                sr = SAVE_SR
                save_one_result(utt, text, wav, sr, fout, audio_out_dir)

            pbar.update(1)
    pbar.close()


def run_audio_mode(args, models):
    in_dir = Path(args.input_path)
    assert in_dir.is_dir(), f"WAV dir not found: {in_dir}"

    wavs = sorted([p for p in in_dir.glob("*.wav")])
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    text_out = outdir / "text_output.jsonl"
    audio_out_dir = outdir / "audio_output"

    pbar = tqdm(total=len(wavs), desc="Batch(audio) infer", ncols=100)
    with open(text_out, "w", encoding="utf-8") as fout:
        for wav_path in wavs:
            try:
                utt = wav_path.stem  # 直接用 basename 作为 utt
                user_audio_tokens = process_audio_to_tokens(
                    str(wav_path), models["whisper_model"], models["feature_extractor"]
                )
                system_prompt = (
                    "User will provide you with a speech instruction. Do it step by step. "
                    "First, think about the instruction and respond in a interleaved manner, "
                    "with 13 text token followed by 26 audio tokens. "
                )
                full_prompt = f"<|system|>\n{system_prompt}<|user|>\n{user_audio_tokens}<|assistant|>streaming_transcription\n"

                # 生成
                out_ids = generate_ids(
                    full_prompt,
                    models["glm_tokenizer"],
                    models["glm_model"],
                    args.device,
                    args.temperature,
                    args.top_p,
                    args.max_new_token,
                )

                # 解析
                text, audio_ids = parse_generation(out_ids, models["end_q"], models["audio_offset"], models["glm_tokenizer"])

                # 声码
                wav, sr = vocode(audio_ids, models["audio_decoder"], args.device)
                if wav is None:
                    fout.write(json.dumps({"utt": utt, "output": text}, ensure_ascii=False) + "\n")
                    fout.flush()
                else:
                    sr = SAVE_SR
                    save_one_result(utt, text, wav, sr , fout, audio_out_dir)

            except Exception as e:
                # 出错不中断
                # 也可在这里写一条错误日志到 jsonl 以便定位
                pass
            finally:
                pbar.update(1)
    pbar.close()


# ----------------------------
# CLI
# ----------------------------
def build_args():
    ap = argparse.ArgumentParser(description="GLM-4-Voice batch inference (text/audio)")
    ap.add_argument("--input-mode", type=str, choices=["audio", "text"], required=True)
    ap.add_argument("--input-path", type=str, required=True, help="text: JSONL; audio: WAV 目录")
    ap.add_argument("--output-dir", type=str, required=True, help="输出目录（内含 text_output.jsonl 与 audio_output/）")

    # text 模式专属
    ap.add_argument("--field", type=str, default="prompt", help="JSONL 中读取的字段名，默认 prompt")

    # 生成超参
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.8)
    ap.add_argument("--max_new_token", type=int, default=2000)

    # 路径与设备
    ap.add_argument("--flow-path", type=str, default="./glm-4-voice-decoder")
    ap.add_argument("--model-path", type=str, default="THUDM/glm-4-voice-9b")
    ap.add_argument("--tokenizer-path", type=str, default="THUDM/glm-4-voice-tokenizer")
    ap.add_argument("--device", type=str, default="cuda")
    return ap.parse_args()


def main():
    args = build_args()
    models = load_models(args)

    if args.input_mode == "text":
        run_text_mode(args, models)
    else:
        run_audio_mode(args, models)


if __name__ == "__main__":
    main()

# -m debugpy --listen 5678 --wait-for-client

# Text mode example:
# HF_ENDPOINT="https://hf-mirror.com" python batch_infer_glm4voice.py \
#   --input-mode text \
#   --input-path /root/data/safety/jsonl/XSTest_test.jsonl \
#   --output-dir /root/data/safety/model_answer/GLM-4-Voice/text_in \
#   --field prompt 


# Audio mode example:
# HF_ENDPOINT="https://hf-mirror.com" python batch_infer_glm4voice.py \
#   --input-mode audio \
#   --input-path /path/to/wav_dir \
#   --output-dir /path/to/outdir