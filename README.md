# Live Bible STT

Offline speech-to-text for Bible verse lookup with OBS output.

## Setup

1. Create/activate your venv.
2. Install deps:

```powershell
python -m pip install -r requirements.txt
```

If you don't have `requirements.txt`, install the essentials:

```powershell
python -m pip install faster-whisper sounddevice numpy
```

## Models

Large model files are intentionally **not** tracked in git. Place them in:

```
whisper_models/
```

`fw_test_stt.py` points to:

```
whisper_models/faster-whisper-turbo
```

If you want a different model, update `MODEL_PATH` in `fw_test_stt.py`.

## Run

```powershell
python fw_test_stt.py
```

OBS output is written to `obs_output.txt`.
