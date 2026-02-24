from flask import Flask, render_template_string
from flask_socketio import SocketIO
import threading
import queue
import sys
import time
import re
import os
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from bible_search import search_bible, is_confident_match, get_next_verses

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# ================= CONFIG =================

MIC_SAMPLE_RATE = 16000
RESET_AFTER_SECONDS = 20       # Raised — prevents same verse refiring too quickly
ANTICIPATION_TIMEOUT = 3
MODEL_PATH = "whisper_models/faster-whisper-small"
CHUNK_SECONDS = 4
OVERLAP_SECONDS = 0.2
MIN_RMS = 0.004
MIN_PEAK = 0.01
MIN_TEXT_CHARS = 6
MIN_WORDS = 2
MIN_SEARCH_WORDS = 3
MAX_VERSES_ON_SCREEN = 4
MAX_TRANSCRIPT_CHARS = 300     # Reject absurdly long hallucinated strings

# ================= STATE =================

state = {
    "running": False,
    "listening": False,
    "verse_fired": False,
    "last_fire_time": 0,
    "continuation_queue": [],
    "continuation_index": 0,
    "current_book_context": None,
    "anticipation_mode": False,
    "anticipation_time": 0,
    "last_fired_verse_key": None,
    "current_verse": None,
    "log": [],
}

audio_queue = queue.Queue()
stt_thread = None

# ================= HELPERS =================

def normalize(text):
    return re.sub(r"[^\w\s]", "", text.lower()).strip()

def is_hallucination(text: str) -> bool:
    """Detect Whisper looping hallucinations like 'I'm sorry, I'm sorry...'"""
    words = text.strip().split()
    if len(words) < 6:
        return False
    # If any 3-word phrase repeats more than 4 times, it's a hallucination
    for i in range(len(words) - 2):
        phrase = " ".join(words[i:i+3])
        if text.count(phrase) > 4:
            return True
    return False

def extract_control_command(text):
    if re.search(r"\b(next|continue|go on)\b", text): return "CONTINUE"
    if re.search(r"\b(stop|pause|clear|reset)\b", text): return "RESET"
    return None

BOOK_NAMES = [
    "genesis","exodus","leviticus","numbers","deuteronomy",
    "psalms","psalm","proverbs","ecclesiastes","isaiah","jeremiah",
    "matthew","mark","luke","john","acts","romans","corinthians",
    "galatians","ephesians","philippians","colossians","thessalonians",
    "timothy","titus","hebrews","james","peter","jude","revelation"
]

def detect_book_context(text):
    for b in BOOK_NAMES:
        if b in text:
            return b.title()
    return None

def log(msg):
    state["log"].append(msg)
    if len(state["log"]) > 100:
        state["log"].pop(0)
    socketio.emit("log", {"message": msg})

def display_verses(verses):
    if not verses:
        return
    first = verses[0]
    last = verses[-1]
    title = f"{first['book']} {first['chapter']}:{first['verse']}–{last['verse']}"
    body = " ".join(v["text"] for v in verses)
    state["current_verse"] = {"title": title.upper(), "body": body}
    socketio.emit("verse", {"title": title.upper(), "body": body})
    log(f"📖 {title}")

def clear_display():
    state["current_verse"] = None
    socketio.emit("verse", {"title": "", "body": ""})

# ================= AUDIO =================

def audio_callback(indata, frames, time_info, status):
    if state["running"]:
        audio_queue.put(indata.copy())

model = WhisperModel(MODEL_PATH, device="cpu", compute_type="int8")

def stt_loop():
    chunk_size = int(MIC_SAMPLE_RATE * CHUNK_SECONDS)
    overlap_size = int(MIC_SAMPLE_RATE * OVERLAP_SECONDS)
    audio_buffer = np.zeros((0, 1), dtype=np.int16)

    with sd.InputStream(
        samplerate=MIC_SAMPLE_RATE,
        channels=1,
        dtype="int16",
        callback=audio_callback,
    ):
        log("🎙️ Microphone open and listening...")
        while state["running"]:
            try:
                chunk = audio_queue.get(timeout=1)
            except queue.Empty:
                continue

            audio_buffer = np.concatenate([audio_buffer, chunk], axis=0)
            if len(audio_buffer) < chunk_size:
                continue

            chunk_audio = audio_buffer[:chunk_size]
            audio_buffer = audio_buffer[chunk_size - overlap_size:]

            audio_float = chunk_audio.astype(np.float32).squeeze() / 32768.0
            rms = float(np.sqrt(np.mean(audio_float ** 2)))
            peak = float(np.max(np.abs(audio_float)))

            socketio.emit("levels", {"rms": round(rms, 5), "peak": round(peak, 5)})

            if rms < MIN_RMS or peak < MIN_PEAK:
                continue

            segments, _ = model.transcribe(
                audio_float,
                language="en",
                vad_filter=True,
                beam_size=3,
                temperature=0.0,
                condition_on_previous_text=False,
                no_speech_threshold=0.7,   # Raised — rejects silence more aggressively
                log_prob_threshold=-0.4,   # Raised — rejects uncertain transcripts
                vad_parameters=dict(
                    min_silence_duration_ms=1000,
                    speech_pad_ms=120,
                    threshold=0.5,
                ),
            )
            text = " ".join(s.text.strip() for s in segments).strip()

            if not text or len(text) < MIN_TEXT_CHARS or len(text.split()) < MIN_WORDS:
                continue

            # Reject hallucinated looping transcripts
            if is_hallucination(text):
                log(f"🚫 Hallucination detected, skipping")
                continue

            # Reject absurdly long transcripts
            if len(text) > MAX_TRANSCRIPT_CHARS:
                log(f"🚫 Transcript too long ({len(text)} chars), skipping")
                continue

            norm = normalize(text)
            log(f"🎤 {text}")
            socketio.emit("transcript", {"text": text})

            # ---------- AUTO RESET ----------
            if state["verse_fired"] and time.time() - state["last_fire_time"] > RESET_AFTER_SECONDS:
                state["verse_fired"] = False
                state["last_fired_verse_key"] = None
                state["continuation_queue"] = []
                state["continuation_index"] = 0
                clear_display()

            # ---------- BOOK ANTICIPATION ----------
            book_ctx = detect_book_context(norm)
            if book_ctx:
                state["current_book_context"] = book_ctx
                state["anticipation_mode"] = True
                state["anticipation_time"] = time.time()

            if state["anticipation_mode"] and time.time() - state["anticipation_time"] > ANTICIPATION_TIMEOUT:
                state["anticipation_mode"] = False
                state["current_book_context"] = None

            # ---------- CONTROL COMMANDS ----------
            if state["verse_fired"]:
                cmd = extract_control_command(norm)
                if cmd == "RESET":
                    state["verse_fired"] = False
                    state["last_fired_verse_key"] = None
                    clear_display()
                elif cmd == "CONTINUE":
                    batch = state["continuation_queue"][
                        state["continuation_index"]: state["continuation_index"] + MAX_VERSES_ON_SCREEN
                    ]
                    if batch:
                        state["continuation_index"] += len(batch)
                        display_verses(batch)
                        state["last_fire_time"] = time.time()
                continue

            # ---------- SEARCH ----------
            if len(norm.split()) < MIN_SEARCH_WORDS:
                continue

            results = search_bible(norm, book_hint=state["current_book_context"])
            if is_confident_match(results, threshold=0.58):
                v = results[0]
                verse_key = (v["book"], v["chapter"], v["verse"])

                if verse_key == state["last_fired_verse_key"]:
                    continue

                state["verse_fired"] = True
                state["last_fire_time"] = time.time()
                state["last_fired_verse_key"] = verse_key
                state["current_book_context"] = None
                state["anticipation_mode"] = False

                state["continuation_queue"] = get_next_verses(
                    v["book"], v["chapter"], v["verse"] + 1, max_verses=20
                )
                state["continuation_index"] = MAX_VERSES_ON_SCREEN - 1
                display_verses([v] + state["continuation_queue"][:MAX_VERSES_ON_SCREEN - 1])
            else:
                if results:
                    log(f"⚠️ No match — Top: {results[0]['book']} {results[0]['chapter']}:{results[0]['verse']} ({results[0]['score']:.2f})")

    log("⏹️ STT stopped.")

# ================= SOCKET EVENTS =================

@socketio.on("start")
def handle_start():
    global stt_thread
    if not state["running"]:
        state["running"] = True
        stt_thread = threading.Thread(target=stt_loop, daemon=True)
        stt_thread.start()
        log("▶️ Started listening")
        socketio.emit("status", {"running": True})

@socketio.on("stop")
def handle_stop():
    state["running"] = False
    log("⏹️ Stopped")
    socketio.emit("status", {"running": False})

@socketio.on("clear")
def handle_clear():
    state["verse_fired"] = False
    state["last_fired_verse_key"] = None
    clear_display()
    log("🗑️ Display cleared")

@socketio.on("next")
def handle_next():
    batch = state["continuation_queue"][
        state["continuation_index"]: state["continuation_index"] + MAX_VERSES_ON_SCREEN
    ]
    if batch:
        state["continuation_index"] += len(batch)
        display_verses(batch)

@socketio.on("connect")
def handle_connect():
    socketio.emit("status", {"running": state["running"]})
    if state["current_verse"]:
        socketio.emit("verse", state["current_verse"])
    for msg in state["log"][-50:]:
        socketio.emit("log", {"message": msg})

# ================= ROUTES =================

@app.route("/")
def control():
    with open("control.html", "r", encoding="utf-8") as f:
        return f.read()

@app.route("/overlay")
def overlay():
    with open("overlay.html", "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    print("Server running at http://localhost:5000")
    print("OBS overlay at  http://localhost:5000/overlay")
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)