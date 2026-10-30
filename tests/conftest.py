import os
import sys
import types
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# sermon_worker imports torch, coqui-tts and pydub at module scope. None of the
# tests need the real libraries, so light stand-ins keep the suite fast and
# runnable without the multi-gigabyte ML stack.
_torch = types.ModuleType("torch")
_torch.cuda = types.SimpleNamespace(empty_cache=lambda: None)
_tts = types.ModuleType("TTS")
_tts_api = types.ModuleType("TTS.api")
_tts_api.TTS = object
for name, module in {
    "torch": _torch,
    "TTS": _tts,
    "TTS.api": _tts_api,
}.items():
    sys.modules.setdefault(name, module)

# pydub is small, so the real one is used when it is installed; the MP3 output
# tests need it (and ffmpeg) and skip otherwise.
try:
    import pydub  # noqa: F401
except ImportError:
    _pydub = types.ModuleType("pydub")
    _pydub.AudioSegment = None
    sys.modules["pydub"] = _pydub


@pytest.fixture
def worker():
    import sermon_worker

    return sermon_worker


@pytest.fixture
def app_module(tmp_path, monkeypatch):
    # app.py creates /data at import time, which is not writable here.
    with mock.patch.object(os, "makedirs"):
        import app
    monkeypatch.setattr(app, "DB_PATH", str(tmp_path / "jobs.db"))
    app.init_db()
    return app


@pytest.fixture
def client(app_module):
    return app_module.app.test_client()
