#!/usr/bin/env python3
"""Generate local TTS cache audio files using MiMo API."""

import argparse
import os
import sys

# Add project root to path so we can import audio.tts.mimo_backend
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio.tts.mimo_backend import TTS_CACHE_MAP, LOCAL_AUDIO_DIR, MiMoTTS


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate local TTS cache WAV files using MiMo API")
    parser.add_argument("--force", action="store_true", help="Overwrite existing cache files")
    parser.add_argument("--api-key", default=None, help="MiMo API key (falls back to MIMO_API_KEY env var)")
    return parser.parse_args()


def generate_cache_files(mimo_tts: MiMoTTS, force: bool):
    """Generate all cache files. Returns (generated, skipped, failed) counts."""
    os.makedirs(LOCAL_AUDIO_DIR, exist_ok=True)
    
    total = len(TTS_CACHE_MAP)
    generated = 0
    skipped = 0
    failed = 0
    
    for i, (phrase, filename) in enumerate(TTS_CACHE_MAP.items(), 1):
        wav_path = os.path.join(LOCAL_AUDIO_DIR, f"{filename}.wav")
        
        if os.path.exists(wav_path) and not force:
            print(f"[{i}/{total}] {phrase} -> {filename}.wav (skipped, exists)")
            skipped += 1
            continue
        
        try:
            audio_bytes = mimo_tts._synthesize(phrase)
            with open(wav_path, "wb") as f:
                f.write(audio_bytes)
            print(f"[{i}/{total}] {phrase} -> {filename}.wav (generated)")
            generated += 1
        except Exception as e:
            print(f"[{i}/{total}] {phrase} -> {filename}.wav (FAILED: {e})")
            failed += 1
    
    return total, generated, skipped, failed


def main():
    args = parse_args()
    
    api_key = args.api_key or os.environ.get("MIMO_API_KEY") or os.environ.get("XIAOMI_MIMO_API_KEY") or ""
    if not api_key:
        print("Error: No API key provided. Use --api-key or set MIMO_API_KEY / XIAOMI_MIMO_API_KEY env var.")
        sys.exit(1)
    
    print(f"Initializing MiMo TTS (async_mode=False for cache generation)...")
    mimo_tts = MiMoTTS(api_key=api_key, async_mode=False)
    
    print(f"Generating {len(TTS_CACHE_MAP)} cache files in: {LOCAL_AUDIO_DIR}")
    if args.force:
        print("Force mode: existing files will be overwritten")
    print()
    
    total, generated, skipped, failed = generate_cache_files(mimo_tts, args.force)
    
    print()
    print("=" * 50)
    print(f"Cache generation complete:")
    print(f"  Total:     {total}")
    print(f"  Generated: {generated}")
    print(f"  Skipped:   {skipped}")
    print(f"  Failed:    {failed}")
    print("=" * 50)
    
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
