#!/usr/bin/env python3
"""Batch voice-clone TTS with mlx-audio (Qwen3-TTS Base).

Loads the model once and synthesizes every line with the same cloned voice.

Usage:
    python clone_tts.py --model mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit \
        --ref ref.wav --ref-text "exact transcript of ref.wav" \
        --language Russian \
        --lines '{"line1.wav": "Text one.", "line2.wav": "Text two."}'

--language takes the English language name supported by Qwen3-TTS (Chinese,
English, Japanese, Korean, German, French, Russian, Portuguese, Spanish,
Italian). If the installed mlx-audio build rejects the kwarg, the call is
retried without it.
"""

import argparse
import json

import numpy as np
import soundfile as sf
from mlx_audio.tts.utils import load_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit")
    ap.add_argument("--ref", required=True, help="reference wav (clean voice, 3-10 s)")
    ap.add_argument("--ref-text", required=True, help="exact transcript of the reference")
    ap.add_argument("--language", default=None, help='e.g. "Russian"')
    ap.add_argument("--lines", required=True,
                    help='JSON object: {"out.wav": "text to speak", ...}')
    args = ap.parse_args()

    lines = json.loads(args.lines)
    if not lines:
        raise SystemExit("--lines parsed to an empty mapping")

    print(f"loading {args.model} ...", flush=True)
    model = load_model(args.model)
    print("model loaded", flush=True)

    for fname, text in lines.items():
        kwargs = dict(text=text, ref_audio=args.ref, ref_text=args.ref_text)
        if args.language:
            kwargs["language"] = args.language
        try:
            results = list(model.generate(**kwargs))
        except TypeError:  # older mlx-audio without language kwarg
            kwargs.pop("language", None)
            results = list(model.generate(**kwargs))
        r = results[0]
        audio = np.asarray(r.audio, dtype=np.float32)
        sr = getattr(r, "sample_rate", 24000)
        sf.write(fname, audio, sr)
        print(f"{fname}: {len(audio) / sr:.2f}s @ {sr} Hz", flush=True)

    print("DONE")


if __name__ == "__main__":
    main()
