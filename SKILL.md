---
name: multilingual-audio-replacement
description: re-dub video/audio into another language by voice cloning
version: 1.1.0
author: vokasug, Hermes Agent
license: MIT
platforms: [macos]
metadata:
  hermes:
    tags: [video, audio, dubbing, voice-cloning, tts, source-separation, translation, mlx, qwen3-tts, roformer, ffmpeg]
    related_skills: [translated-video-subtitles, mlx-whisper, yt-dlp]
---

# Multilingual Audio Replacement

End-to-end re-dubbing of a video or audio file into another language: transcribe the
original speech, separate voice from background with a MelBand RoFormer model (music
and ambience stay untouched), translate the lines, synthesize them with a voice cloned
from the original speaker (Qwen3-TTS via mlx-audio), and mix the result back onto the
original timeline. Everything runs locally on Apple Silicon; no cloud services.

Works for video (MP4/MOV, video stream is copied without re-encoding) and for
audio-only files (WAV/MP3/M4A).

## When to Use

- "Переозвучь это видео на русский / английский / ..." keeping the original background
- Replace speech in a clip with the SAME voice speaking another language
- Replace speech with a DIFFERENT provided voice — e.g. a native target-language
  speaker sample (often sounds more natural than a cross-lingual clone)
- Produce a voice-free background track (karaoke-style) from any recording
- Don't use for: burned-in subtitles without audio replacement (use
  `translated-video-subtitles`), or simple transcription only (use `mlx-whisper`).

## Prerequisites (clean Mac setup)

STT is provided by the `mlx-whisper` skill — install it first per its README
(https://github.com/vokasug/hermes-skill-mlx-whisper): the `mlx-whisper` uv tool,
local model folders (ru → `whisper-podlodka-turbo-MLX-q8`, all other languages →
`whisper-large-v3-turbo-8bit`), and the Silero VAD venv. Do NOT pip-install
mlx-whisper into the project venv below — the STT steps call the skill's own
environments.

The rest goes into ONE project venv (Python 3.12). Run installs with the `terminal`
tool; model downloads happen on first use and take several GB — run first-time
commands in the background with `notify=true`.

```bash
# 1. System tools
brew install ffmpeg uv            # if missing

# 2. Project venv (one per task workspace is fine)
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
    mlx-audio soundfile \
    audio-separator onnxruntime audioread
# NOTE: audio-separator forgets to declare onnxruntime and audioread — install them
# explicitly or the CLI crashes with ModuleNotFoundError on first run.

# 3. Models
#    STT:  local folders from the mlx-whisper skill install (nothing to download here)
#    SEP:  vocals_mel_band_roformer.ckpt       (audio-separator downloads it itself)
#    TTS:  mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit (auto-download on first use)
```

## Procedure

1. **Probe the input.** `ffprobe` duration, streams, resolution. Extract audio for
   processing:
   `ffmpeg -y -v error -i INPUT -vn -ac 2 -ar 44100 -c:a pcm_s16le full_audio.wav`
   Completion: duration matches the source; wav exists and is non-empty.

2. **Transcribe via the mlx-whisper skill, and never trust language autodetect.**
   Autodetect can confidently hallucinate the wrong language (observed: a Chinese
   clip "detected" as English with garbage text).
   - **Language gate.** Take the language from a nameable source: title/description
     of the content, the user, or metadata. If the source is uninformative, detect
     on the first 30 s with the raw CLI, WITHOUT `--language`, and read `language`
     from the result:
     `~/.local/bin/mlx_whisper --model ~/.local/share/models/whisper-large-v3-turbo-8bit --clip-timestamps "30" --output-format json --output-dir /tmp/stt-det full_audio.wav`
     You must be able to name WHERE the language came from; never default from
     habit ("it's usually Russian"). A wrong language corrupts ref_text and the
     whole clone.
   - **Main pass — the skill's VAD pipeline** (model auto-selected by language:
     ru → podlodka q8, everything else → large-v3-turbo-8bit):
     ```bash
     ~/.local/share/uv/tools/mlx-whisper/bin/python \
       ~/.hermes/skills/media/mlx-whisper/scripts/vad_transcribe.py full_audio.wav \
       --language <code> --no-correct --debug-segments
     ```
     Consume `~/result-mlx-whisper/YYYY-MM-DD_full_audio.segments.json` — one
     `{start, end, text, logprob}` per speech segment (seconds). The MD next to it
     is a by-product; ignore it.
   - **When in doubt, cross-check with a second model** (e.g. the fp16 folder via
     `--model`): agreement between two models is the strongest signal; otherwise
     pick the pass with more segments, more plausible text, better avg_logprob.
   - **Songs and non-speech:** the pipeline's Silero VAD is a speech/non-speech
     filter — singing yields 0 segments (measured: 59 s of a song → 0). A
     music-only clip therefore means "nothing to dub": deliver the original, that
     is the correct outcome, not a failure. Very quiet speech can theoretically be
     dropped too — the two-model cross-check covers that case.
   Completion: segment list with start/end for every speech line, language confirmed.

3. **Separate voice from background.**
   `.venv/bin/audio-separator -m vocals_mel_band_roformer.ckpt --output_dir sep --output_format WAV full_audio.wav`
   Produces `full_audio_(vocals)_....wav` and `full_audio_(other)_....wav`.
   The `(other)` stem is the background WITHOUT speech — this is the whole point:
   no ducking, no muted holes in the music. Verify by transcribing `(other)` with
   the SOURCE language: it must come back empty. Use the RAW CLI for this check —
   never vad_transcribe: its VAD silently drops all non-speech, so the emptiness
   check would pass vacuously:
   ```bash
   # raw verification transcription (model folder by language: ru →
   # whisper-podlodka-turbo-MLX-q8, any other → whisper-large-v3-turbo-8bit):
   ~/.local/bin/mlx_whisper --model ~/.local/share/models/whisper-large-v3-turbo-8bit \
     --language <code> --condition-on-previous-text False \
     --output-format json --output-dir /tmp/stt-verify "<wav>"
   ```
   Completion: `(other)` stem transcribes to nothing in the source language.

4. **Build the cloning reference.** Two options:
   - **Clone the original speaker** — cut the reference from the `(vocals)` stem
     (never from the raw mix — music bleed ruins the clone). Cut each speech
     segment with ~0.2–0.25 s padding (VAD-path segment starts can lag the true
     onset by up to ~0.6 s; a 0.1 s padding clips word attacks), concat in order,
     boost level, resample to 24 kHz mono. `ref_text` must be the EXACT
     transcript of the concatenated
     segments, in the same order, in the SOURCE language. A wrong ref_text
     silently degrades the clone.
   - **Clone a provided voice sample** (often the better result): a 3-10 s
     recording of a NATIVE speaker of the TARGET language gives more natural
     prosody, rhythm and phonetics than a cross-lingual clone of the source
     speaker. Transcribe the sample with the RAW CLI (forced language, word
     timestamps — never vad_transcribe: its VAD can silently drop quiet speech
     in a short sample and desync ref_text from the audio), cut a
     clean 3-10 s span, resample to 24 kHz mono; `ref_text` = exact transcript
     of that span in the TARGET language. Works for multiple voices in one
     task: keep the same translations, re-synthesize per voice, deliver one
     output file per voice.
   ```bash
   # reference from the (vocals) stem:
   ffmpeg -y -v error -i "sep/full_audio_(vocals)_vocals_mel_band_roformer.wav" \
     -filter_complex "[0:a]atrim=START1:END1,asetpts=PTS-STARTPTS[s1];\
[0:a]atrim=START2:END2,asetpts=PTS-STARTPTS[s2];\
[s1][s2]concat=n=2:v=0:a=1,volume=2.0[out]" \
     -map "[out]" -ac 1 -ar 24000 ref.wav
   ```
   Completion: ref.wav is 3-10 s of clean speech; transcript verified against it.

5. **Group segments into sentences, then translate.** Whisper segments are
   pause-based FRAGMENTS of sentences; dubbing each fragment as its own TTS line
   leaves audible gaps in the middle of a thought and chops phrase endings (a
   fragment can end on a stressed word). Merge consecutive segments into whole
   sentences (group by ending punctuation `.?!`), then translate each sentence
   as ONE line. `split_into_sentences` from the mlx-whisper skill's
   `vad_transcribe.py` does exactly this grouping (word stream cut at terminal
   punctuation, interpolated sentence starts) — import it with the same
   importlib one-liner the mlx-whisper SKILL.md shows for its preflight instead
   of re-implementing. A sentence's time window runs from its first segment's
   start to the next sentence's start (or clip end) — these windows are large,
   so whole sentences fit without time-stretching. If a line is too long,
   shorten the
   translation rather than overlapping the next line. Keep line count = source
   sentence count.
   Completion: every line's estimated duration fits its window.

6. **Synthesize with the cloned voice.** Use `scripts/clone_tts.py` (one process,
   model loaded once for all lines):
   ```bash
   .venv/bin/python <SKILL_DIR>/scripts/clone_tts.py \
     --model mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit \
     --ref ref.wav --ref-text "<exact source transcript>" \
     --language Russian \
     --lines '{"line1.wav": "Первая фраза.", "line2.wav": "Вторая фраза."}'
   ```
   `--language` takes the English language name ("Russian", "English", ...);
   Qwen3-TTS supports Chinese, English, Japanese, Korean, German, French, Russian,
   Portuguese, Spanish, Italian. If the installed mlx-audio version rejects the kwarg
   (TypeError), the script retries without it.
   Completion: one wav per line at the model's sample rate (24 kHz).

7. **Trim TTS edge silence and check fit.** TTS output has lead/tail padding.
   Trim the START hard but the TAIL gently: a symmetric aggressive cut clips the
   natural decay of the last word (vowel tails, breath) and phrases sound chopped
   off. Cut the tail only after 150 ms below -50 dB and add a 60 ms fade-out
   (fade-in between two areverse) so every line ends smoothly:
   ```bash
   ffmpeg -y -v error -i lineN.wav -af \
     "silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.05,\
areverse,silenceremove=start_periods=1:start_threshold=-50dB:start_silence=0.15,afade=t=in:d=0.06,areverse" \
     lineN_trim.wav
   ```
   `ffprobe` each trimmed file: lineN must fit its window from step 5. If not,
   shorten the text and regenerate (do NOT time-stretch the voice). Different
   voices speak at different speeds: lines that fit with one cloned voice can
   overflow with another — re-run the fit check for EVERY voice and shorten
   per voice.
   Completion: all trimmed durations fit.

8. **Mix background + cloned lines at the ORIGINAL timestamps.**
   ```bash
   ffmpeg -y -v error -i INPUT -i "sep/full_audio_(other)_vocals_mel_band_roformer.wav" \
     -i line1_trim.wav -i line2_trim.wav -filter_complex "
   [1:a]apad=whole_dur=DUR_PLUS_1[bg];
   [2:a]adelay=START1_MS|START1_MS[d1];
   [3:a]adelay=START2_MS|START2_MS[d2];
   [bg][d1][d2]amix=inputs=3:duration=first:normalize=0,alimiter=limit=0.95,atrim=duration=DUR[aout]" \
     -map 0:v -map "[aout]" -c:v copy -c:a aac -b:a 128k OUT.mp4
   ```
   For audio-only input drop `-map 0:v -c:v copy` and encode to the input's format.
   The `(other)` stem replaces the original audio entirely; lines are delayed to the
   exact millisecond where the source speech started. If the cloned voice is quieter
   than the background, add `volume=1.3` (etc.) to each TTS input, not to the bg.
   Completion: output duration ≈ source duration (±0.1 s).

9. **Verify before delivery** (all via `terminal`):
   - `ffprobe` output: duration ≈ source, expected codecs, video stream identical
     dimensions (`-c:v copy` must have been used for video).
   - Full decode: `ffmpeg -v error -i OUT -f null -` → no output.
   - Transcribe the FINAL output in the TARGET language (raw CLI from step 3,
     target-language model folder): every translated line present, in order.
   - Transcribe the FINAL output in the SOURCE language (raw CLI): no source
     speech left (music may produce stray hallucinations — judge by coherence,
     not emptiness).
   - Deliver via `MEDIA:/absolute/path` (Telegram: H.264+AAC MP4 < 50 MB).

## Pitfalls

- **Bare `apad` creates an infinite stream.** `amix duration=first` with a padded
  first input then encodes forever (a 30 s clip once produced a 33-hour, 64 MB
  file). ALWAYS bound padding with `apad=whole_dur=N` and/or a final `atrim`.
- **Never duck or mute when you have stems.** Volume envelopes around speech punch
  audible holes in the music; the `(other)` stem keeps the background continuous.
- **Language autodetect lies.** Gate the language on a nameable source or a 30 s
  detect, then force `--language`; cross-check with a second model when unsure.
  A wrong language choice corrupts ref_text and the whole clone.
- **Never run verification STT through vad_transcribe.** Its VAD drops singing and
  any non-speech (measured: 59 s of a song → 0 intervals): an `(other)`-stem
  emptiness check would pass vacuously, and a quiet voice sample could lose
  speech and desync ref_text. All verification and ref-sample transcription =
  raw mlx_whisper CLI only. Conversely, 0 segments from the MAIN pass on a
  music-only clip is a correct "nothing to dub", not an error.
- **Clone reference must be clean isolated speech**, cut tightly at STT
  timestamps, with a transcript that matches word-for-word — either from the
  `(vocals)` stem (original speaker) or from a user-provided sample (native
  target-language voice, frequently the more natural-sounding choice).
- **Dub sentences, not Whisper segments.** Segments are pause-based fragments;
  one TTS line per fragment produces gaps mid-thought and clipped phrase endings.
  Group segments into sentences before translating (step 5).
- **Hard tail trims clip word endings.** `silenceremove` at -45 dB/0.05 s on the
  tail cuts the decay of the final syllable (observed: «...знаете что» lost the
  final «о»). Use the gentle tail threshold + fade-out from step 7.
- **Trim TTS silence before measuring fit**, and place lines by their trimmed start,
  not the raw file start.
- **First run downloads several GB of models** (RoFormer ~0.2 GB,
  Qwen3-TTS-8bit ~2 GB; the whisper models ship with the mlx-whisper skill
  install). Run with `background=true, notify=true`.
- MLX is Apple-Silicon-only; this skill does not work on Intel Macs or Linux/Windows.
- Keep `c:v copy` for video: the task is audio replacement, re-encoding the picture
  is pure loss.

## Verification

- Output duration matches source within 0.1 s; full ffmpeg decode is clean.
- Target-language transcription of the output contains exactly the translated lines.
- Source-language transcription of the `(other)` stem is empty; of the output — no
  coherent source speech.
- Video stream was copied (ffprobe shows original codec/profile, encode took seconds).
