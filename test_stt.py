import json
import pyaudio
from vosk import Model, KaldiRecognizer
from collections import deque
import time


# Path to your Vosk model
MODEL_PATH = "models/vosk-model-small-en-us"

# Load model
model = Model(MODEL_PATH)
recognizer = KaldiRecognizer(model, 16000)
recognizer.SetWords(True)


# Rolling buffer: (timestamp, text)
text_buffer = deque(maxlen=10)  # ~last few seconds


# Audio setup
p = pyaudio.PyAudio()
stream = p.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=16000,
    input=True,
    frames_per_buffer=4000
)

stream.start_stream()

print("🎤 Listening... Speak into the microphone")

try:
    while True:
        data = stream.read(4000, exception_on_overflow=False)

        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            text = result.get("text", "").strip()

            if text:
                text_buffer.append((time.time(), text))
                print(f"[FINAL] {text}")

        else:
            partial = json.loads(recognizer.PartialResult())
            partial_text = partial.get("partial", "")
            if partial_text:
                print(f"[PARTIAL] {partial_text}", end="\r")

except KeyboardInterrupt:
    print("\n🛑 Stopped")

finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
