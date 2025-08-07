import argparse
import os
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm
from transformers import WhisperFeatureExtractor
from speech_tokenizer.modeling_whisper import WhisperVQEncoder
from speech_tokenizer.utils import extract_speech_token
from flow_inference import AudioDecoder
import sys

sys.path.insert(0, os.path.abspath('third_party/Matcha-TTS'))
sys.path.insert(0, "./cosyvoice")

def reconstruct_audio(input_wav, output_wav, whisper_model, feature_extractor, audio_decoder, device):
    tokens = extract_speech_token(whisper_model, feature_extractor, [input_wav])[0]
    if not tokens:
        print(f"❌ No tokens extracted: {input_wav}")
        return False

    token_tensor = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    audio, _ = audio_decoder.token2wav(token_tensor, uuid=None, finalize=True)

    torchaudio.save(output_wav, audio.cpu(), 22050)
    return True

def main(args):
    device = args.device

    # 加载模型
    whisper_model = WhisperVQEncoder.from_pretrained(args.tokenizer_path).eval().to(device)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.tokenizer_path)
    audio_decoder = AudioDecoder(
        config_path=f"{args.flow_path}/config.yaml",
        flow_ckpt_path=f"{args.flow_path}/flow.pt",
        hift_ckpt_path=f"{args.flow_path}/hift.pt",
        device=device
    )

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    wav_paths = list(input_dir.rglob("*.flac")) + list(input_dir.rglob("*.wav"))
    print(f"🔍 Found {len(wav_paths)} audio files in {input_dir}")

    for wav_path in tqdm(wav_paths, desc="Reconstructing audio"):
        output_filename = wav_path.stem + ".wav"
        out_path = output_dir / output_filename

        try:
            success = reconstruct_audio(str(wav_path), str(out_path), whisper_model, feature_extractor, audio_decoder, device)
            if not success:
                tqdm.write(f"⚠️ Failed: {wav_path}")
        except Exception as e:
            tqdm.write(f"❗ Error processing {wav_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True, help="Input directory containing .wav files.")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory to save reconstructed wav files.")
    parser.add_argument("--tokenizer-path", type=str, default="THUDM/glm-4-voice-tokenizer")
    parser.add_argument("--flow-path", type=str, default="./glm-4-voice-decoder")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    main(args)

# python batch_reconstruct_audio.py --input-dir /root/data/LibriSpeech/test-clean --output-dir /root/data/LibriSpeech/GLM-4-Voice-rec/wav --tokenizer-path THUDM/glm-4-voice-tokenizer --flow-path ./glm-4-voice-decoder --device cuda