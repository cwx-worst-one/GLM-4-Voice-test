import argparse
import torch
import torchaudio
from transformers import WhisperFeatureExtractor
from speech_tokenizer.modeling_whisper import WhisperVQEncoder
from speech_tokenizer.utils import extract_speech_token
from flow_inference import AudioDecoder
import os, sys
sys.path.insert(0, os.path.abspath('third_party/Matcha-TTS'))
sys.path.insert(0, "./cosyvoice")
sys.path.insert(0, "./third_party/Matcha-TTS")

def main(args):
    device = args.device
    whisper_model = WhisperVQEncoder.from_pretrained(args.tokenizer_path).eval().to(device)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.tokenizer_path)
    audio_decoder = AudioDecoder(
        config_path=f"{args.flow_path}/config.yaml",
        flow_ckpt_path=f"{args.flow_path}/flow.pt",
        hift_ckpt_path=f"{args.flow_path}/hift.pt",
        device=device
    )

    tokens = extract_speech_token(whisper_model, feature_extractor, [args.input_wav])[0]
    if not tokens:
        raise ValueError("No tokens extracted from audio.")
    else:
        print(f"Extracted {len(tokens)} tokens from the audio.")

    token_tensor = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    audio, _ = audio_decoder.token2wav(token_tensor, uuid=None, finalize=True)

    torchaudio.save(args.output_wav, audio.cpu(), 22050)
    print(f"✅ Reconstructed audio saved to {args.output_wav}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-wav", type=str, required=True, help="Path to input wav file.")
    parser.add_argument("--output-wav", type=str, required=True, help="Path to save reconstructed wav.")
    parser.add_argument("--tokenizer-path", type=str, default="THUDM/glm-4-voice-tokenizer")
    parser.add_argument("--flow-path", type=str, default="./glm-4-voice-decoder")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    main(args)

# python reconstruct_audio_from_vq.py --input-wav ./example.flac --output-wav ./reconstructed.wav --tokenizer-path THUDM/glm-4-voice-tokenizer --flow-path ./glm-4-voice-decoder --device cuda