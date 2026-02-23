import queue
import sys
import time
import re
import os
import numpy as np
import textwrap
import sounddevice as sd

from faster_whisper import WhisperModel

from bible_search import (
    search_bible,
    is_confident_match,
    get_next_verses,
)

# ================= OBS OUTPUT =================

MAX_LINE_WIDTH = 48
MAX_VERSES_ON_SCREEN = 4

def write_to_obs(title: str, body: str):
    with open("obs_output.txt", "w", encoding="utf-8") as f:
        f.write(title.strip() + "\n\n" + body.strip())
        f.flush()
        os.fsync(f.fileno())

def format_for_obs(title: str, body: str):
    return title.upper(), "\n".join(textwrap.wrap(body, MAX_LINE_WIDTH))

def display_verses(verses):
    if not verses:
        return
    first = verses[0]
    last = verses[-1]
    title = f"{first['book']} {first['chapter']}:{first['verse']}–{last['verse']}"
    body = " ".join(v["text"] for v in verses)
    title, body = format_for_obs(title, body)
    write_to_obs(title, body)
    print("\n📖 DISPLAY")
    print(title)
    print(body)
    print("-" * 60)

def clear_display():
    write_to_obs("", "")

# ================= CONFIG =================

MIC_SAMPLE_RATE = 16000
RESET_AFTER_SECONDS = 6
SCRIPTURE_MODE_TIMEOUT = 8
ANTICIPATION_TIMEOUT = 3
MODEL_PATH = "whisper_models/faster-whisper-base"
CHUNK_SECONDS = 4
OVERLAP_SECONDS = 0.2
MIN_RMS = 0.004
MIN_PEAK = 0.01
MIN_TEXT_CHARS = 6
MIN_WORDS = 2
MIN_SEARCH_WORDS = 4        # Minimum words before attempting Bible search
PRINT_RMS_DEBUG = True

# ================= NORMALIZATION =================

def normalize(text: str):
    return re.sub(r"[^\w\s]", "", text.lower()).strip()

# ================= COMMANDS =================

def extract_control_command(text):
    if re.search(r"\b(next|continue|go on)\b", text): return "CONTINUE"
    if re.search(r"\b(stop|pause|clear|reset)\b", text): return "RESET"
    return None

# ================= SCRIPTURE INTENT =================

SCRIPTURE_INTENT_PATTERNS = [
    r"\bthe bible says\b",
    r"\bscripture says\b",
    r"\bas it is written\b",
    r"\bword of god says\b",
    r"\bopen to\b",
]

def detect_scripture_intent(text):
    return any(re.search(p, text) for p in SCRIPTURE_INTENT_PATTERNS)

# ================= BOOK ANTICIPATION =================

BOOK_NAMES = [
    "genesis","exodus","leviticus","numbers","deuteronomy",
    "psalms","psalm","proverbs","ecclesiastes","isaiah","jeremiah",
    "matthew","mark","luke","john","acts","romans",
    "corinthians","galatians","ephesians","philippians",
    "colossians","thessalonians","timothy","titus","hebrews",
    "james","peter","jude","revelation"
]

def detect_book_context(text):
    for b in BOOK_NAMES:
        if b in text:
            return b.title()
    return None

# ================= STT MODEL =================

model = WhisperModel(MODEL_PATH, device="cpu", compute_type="int8")

audio_queue = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if status:
        print(status, file=sys.stderr)
    audio_queue.put(indata.copy())

# ================= MAIN LOOP =================

def run():
    verse_fired = False
    last_fire_time = 0
    continuation_queue = []
    continuation_index = 0
    scripture_mode = False
    scripture_mode_time = 0
    anticipation_mode = False
    anticipation_time = 0
    current_book_context = None
    last_fired_verse_key = None

    chunk_size = int(MIC_SAMPLE_RATE * CHUNK_SECONDS)
    overlap_size = int(MIC_SAMPLE_RATE * OVERLAP_SECONDS)
    audio_buffer = np.zeros((0, 1), dtype=np.int16)

    print("Listening (faster-whisper offline)...")

    with sd.InputStream(
        samplerate=MIC_SAMPLE_RATE,
        channels=1,
        dtype="int16",
        callback=audio_callback,
    ):
        while True:
            chunk = audio_queue.get()
            audio_buffer = np.concatenate([audio_buffer, chunk], axis=0)

            if len(audio_buffer) < chunk_size:
                continue

            chunk_audio = audio_buffer[:chunk_size]
            audio_buffer = audio_buffer[chunk_size - overlap_size:]

            audio_float = chunk_audio.astype(np.float32).squeeze() / 32768.0
            rms = float(np.sqrt(np.mean(audio_float ** 2)))
            peak = float(np.max(np.abs(audio_float)))

            if PRINT_RMS_DEBUG:
                print(f"[RMS] {rms:.5f}  [PEAK] {peak:.5f}")

            if rms < MIN_RMS or peak < MIN_PEAK:
                continue

            segments, _ = model.transcribe(
                audio_float,
                language="en",
                vad_filter=True,
                beam_size=3,
                temperature=0.0,
                condition_on_previous_text=False,
                no_speech_threshold=0.6,
                log_prob_threshold=-0.5,
                vad_parameters=dict(
                    min_silence_duration_ms=1000,
                    speech_pad_ms=120,
                    threshold=0.5,
                ),
            )
            text = " ".join(s.text.strip() for s in segments).strip()

            if not text:
                continue
            if len(text) < MIN_TEXT_CHARS:
                continue
            if len(text.split()) < MIN_WORDS:
                continue

            norm = normalize(text)
            print(f"[STT] {text}")

            # ---------- AUTO RESET ----------
            if verse_fired and time.time() - last_fire_time > RESET_AFTER_SECONDS:
                verse_fired = False
                last_fired_verse_key = None
                continuation_queue.clear()
                continuation_index = 0
                clear_display()

            # ---------- BOOK ANTICIPATION ----------
            book_ctx = detect_book_context(norm)
            if book_ctx:
                current_book_context = book_ctx
                anticipation_mode = True
                anticipation_time = time.time()

            if anticipation_mode and time.time() - anticipation_time > ANTICIPATION_TIMEOUT:
                anticipation_mode = False
                current_book_context = None

            # ---------- SCRIPTURE INTENT ----------
            if detect_scripture_intent(norm):
                scripture_mode = True
                scripture_mode_time = time.time()

            if scripture_mode and time.time() - scripture_mode_time > SCRIPTURE_MODE_TIMEOUT:
                scripture_mode = False

            # ---------- CONTROL ----------
            if verse_fired:
                cmd = extract_control_command(norm)
                if cmd == "RESET":
                    verse_fired = False
                    last_fired_verse_key = None
                    clear_display()
                elif cmd == "CONTINUE":
                    batch = continuation_queue[
                        continuation_index: continuation_index + MAX_VERSES_ON_SCREEN
                    ]
                    if batch:
                        continuation_index += len(batch)
                        display_verses(batch)
                        last_fire_time = time.time()
                continue

            # ---------- SEARCH ----------
            # Skip very short phrases — too likely to false match
            if len(norm.split()) < MIN_SEARCH_WORDS:
                print(f"[SKIP] Too short to search: '{norm}'")
                continue

            results = search_bible(norm, book_hint=current_book_context)
            if is_confident_match(results, threshold=0.68):
                v = results[0]
                verse_key = (v["book"], v["chapter"], v["verse"])

                if verse_key == last_fired_verse_key:
                    print(f"[SKIP] Same verse as last fired: {verse_key}")
                    continue

                verse_fired = True
                last_fire_time = time.time()
                last_fired_verse_key = verse_key
                scripture_mode = False
                anticipation_mode = False
                current_book_context = None

                continuation_queue = get_next_verses(
                    v["book"], v["chapter"], v["verse"] + 1, max_verses=20
                )
                continuation_index = MAX_VERSES_ON_SCREEN - 1
                display_verses(
                    [v] + continuation_queue[:MAX_VERSES_ON_SCREEN - 1]
                )
            else:
                # Show top result in terminal so you can debug what's matching
                if results:
                    print(f"[NO MATCH] Top: {results[0]['book']} {results[0]['chapter']}:{results[0]['verse']} score={results[0]['score']:.3f}")

if __name__ == "__main__":
    run()