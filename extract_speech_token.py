# extract_speech_token.py
import argparse
from transformers import WhisperFeatureExtractor
from speech_tokenizer.modeling_whisper import WhisperVQEncoder
from speech_tokenizer.utils import extract_speech_token

def load_whisper_tokenizer(tokenizer_path: str, device: str):
    model = WhisperVQEncoder.from_pretrained(tokenizer_path).eval().to(device)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(tokenizer_path)
    return model, feature_extractor

def main(args):
    whisper_model, feature_extractor = load_whisper_tokenizer(args.tokenizer_path, args.device)
    tokens = extract_speech_token(whisper_model, feature_extractor, [args.audio_path])[0]
    
    if not tokens:
        print("❌ No speech tokens extracted.")
        return
    
    # import soundfile as sf
    # audio = sf.read(args.audio_path, dtype='float32')[0]
    # print(f"The duration of the audio is {len(audio) / 16000:.2f} seconds.")
    print(f"Extracted {len(tokens)} speech tokens from the audio.")
    print("✅ Extracted speech tokens:")
    print(tokens)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract speech tokens from an audio file.")
    parser.add_argument("--audio-path", type=str, required=True, help="Path to input audio file.")
    parser.add_argument("--tokenizer-path", type=str, default="THUDM/glm-4-voice-tokenizer", help="Path or name of tokenizer.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (e.g., cuda or cpu).")
    args = parser.parse_args()

    main(args)

# python extract_speech_token.py --audio-path /root/data/LibriSpeech/test-clean/61/70968/61-70968-0000.flac --tokenizer-path THUDM/glm-4-voice-tokenizer --device cuda

# python -m debugpy --listen 5678 --wait-for-client extract_speech_token.py --audio-path /root/data/LibriSpeech/test-clean/61/70968/61-70968-0000.flac --tokenizer-path THUDM/glm-4-voice-tokenizer --device cuda