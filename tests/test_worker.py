import math
import shutil
import sqlite3
import struct
import sys
import wave
from pathlib import Path

import pytest
from helpers import run_on

# Longest text a single tts_to_file() call may receive (text_splitter's
# MAX_SEGMENT_CHARS). Written out so these tests also run against code that
# predates that module.
SAFE_MAX = 400

# POST /sermon only accepts a UUID for sermon_guid.
GUID = "1b9d6bcd-bbfd-4b2d-9b5d-ab8dfbbd4bed"


class FakeTTS:
    """Records every tts_to_file() call and fails the way VITS does on empty input."""

    def __init__(self, fail_on=None):
        self.calls = []
        self.fail_on = fail_on

    def tts_to_file(self, text, file_path):
        self.calls.append(text)
        if not text.strip():
            raise RuntimeError("cannot synthesize empty text")
        if self.fail_on is not None and len(self.calls) == self.fail_on:
            raise RuntimeError("synthesis failed")
        with open(file_path, "wb") as f:
            f.write(b"audio")


@pytest.fixture
def combined(worker, monkeypatch):
    joined = []

    def fake_combine(files, out):
        joined.append(list(files))
        with open(out, "wb") as f:
            f.writelines(Path(part).read_bytes() for part in files)

    monkeypatch.setattr(worker, "combine_audio_files", fake_combine)
    return joined


def test_synthesize_text_leading_long_sentence_succeeds(worker, combined, tmp_path):
    tts = FakeTTS()
    text = run_on(1107) + ". Hola a todos. Espero que estén bien."
    worker.synthesize_text(tts, text, str(tmp_path / "out.mp3"))
    assert tts.calls
    assert all(c.strip() and len(c) <= SAFE_MAX for c in tts.calls)
    assert combined and len(combined[0]) == len(tts.calls)


def test_synthesize_text_run_on_paragraph_bounded_segments(worker, combined, tmp_path):
    tts = FakeTTS()
    text = run_on(6526)
    worker.synthesize_text(tts, text, str(tmp_path / "out.mp3"))
    assert all(c.strip() and len(c) <= SAFE_MAX for c in tts.calls)
    assert " ".join(tts.calls) == text


@pytest.mark.parametrize("text", ["", "   \n\t "])
def test_synthesize_text_whitespace_only_raises_without_calling_tts(worker, combined, tmp_path, text):
    tts = FakeTTS()
    with pytest.raises(ValueError):
        worker.synthesize_text(tts, text, str(tmp_path / "out.mp3"))
    assert tts.calls == []


def test_synthesize_text_failed_segment_removes_temp_files(worker, combined, tmp_path):
    tts = FakeTTS(fail_on=3)
    with pytest.raises(RuntimeError):
        worker.synthesize_text(tts, "Hola a todos, hermanos y hermanas. " * 200, str(tmp_path / "out.mp3"))
    assert list(tmp_path.iterdir()) == []


def insert_job(db_path, guid, status="pending", transcription="Hola a todos."):
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO sermons (sermon_guid, transcription, status) VALUES (?, ?, ?)",
        (guid, transcription, status),
    )
    conn.commit()
    conn.close()


def fetch_job(db_path, guid):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM sermons WHERE sermon_guid = ?", (guid,)).fetchone()
    conn.close()
    return row


@pytest.fixture
def model(worker, app_module, combined, monkeypatch, tmp_path):
    """Point the worker at the test database and hand it a FakeTTS."""
    audio_dir = tmp_path / "audiofiles"
    audio_dir.mkdir()
    tts = FakeTTS()
    monkeypatch.setattr(worker, "DB_PATH", app_module.DB_PATH)
    monkeypatch.setattr(worker, "AUDIO_DIR", str(audio_dir))
    monkeypatch.setattr(worker.TTSModelSingleton, "get_instance", classmethod(lambda cls: tts))
    return tts


def set_finished_at(db_path, guid, finished_at):
    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE sermons SET finished_at = ? WHERE sermon_guid = ?", (finished_at, guid))
    conn.commit()
    conn.close()


def test_process_pending_jobs_success_marks_complete(worker, app_module, model):
    insert_job(app_module.DB_PATH, "g-ok", transcription="Hola a todos.")

    worker.process_pending_jobs()

    job = fetch_job(app_module.DB_PATH, "g-ok")
    assert job["status"] == "complete"
    assert job["finished_at"] is not None
    assert job["audio_file"].endswith("g-ok.mp3")
    assert Path(job["audio_file"]).read_bytes() == b"audio"
    assert model.calls == ["Hola a todos."]


def test_process_pending_jobs_error_sets_finished_at(worker, app_module, model):
    insert_job(app_module.DB_PATH, "g-1", transcription="   ")

    worker.process_pending_jobs()

    job = fetch_job(app_module.DB_PATH, "g-1")
    assert job["status"] == "error"
    assert job["finished_at"] is not None


def test_process_pending_jobs_does_not_overwrite_a_row_that_left_pending(worker, app_module, model, monkeypatch):
    insert_job(app_module.DB_PATH, "g-gone")

    def synthesize_then_row_changes(tts, text, output_file):
        conn = sqlite3.connect(app_module.DB_PATH)
        conn.execute("UPDATE sermons SET status = 'error' WHERE sermon_guid = 'g-gone'")
        conn.commit()
        conn.close()

    monkeypatch.setattr(worker, "synthesize_text", synthesize_then_row_changes)
    worker.process_pending_jobs()

    job = fetch_job(app_module.DB_PATH, "g-gone")
    assert job["status"] == "error"
    assert job["audio_file"] is None


def test_purge_old_jobs_deletes_error_row_once_finished(worker, app_module, model):
    insert_job(app_module.DB_PATH, "g-old", status="error")
    insert_job(app_module.DB_PATH, "g-recent", status="error")
    set_finished_at(app_module.DB_PATH, "g-old", "2026-01-01 00:00:00")
    set_finished_at(app_module.DB_PATH, "g-recent", "2999-01-01 00:00:00")

    worker.purge_old_jobs()

    assert fetch_job(app_module.DB_PATH, "g-old") is None
    assert fetch_job(app_module.DB_PATH, "g-recent") is not None


def test_purge_endpoint_uses_worker_purge(client, app_module, model):
    insert_job(app_module.DB_PATH, "g-old", status="complete")
    set_finished_at(app_module.DB_PATH, "g-old", "2026-01-01 00:00:00")

    assert client.post("/purge").status_code == 200
    assert fetch_job(app_module.DB_PATH, "g-old") is None


def test_init_db_stamps_legacy_error_rows_so_they_purge(app_module):
    insert_job(app_module.DB_PATH, "g-legacy", status="error")
    insert_job(app_module.DB_PATH, "g-pending")

    app_module.init_db()
    app_module.init_db()  # idempotent

    assert fetch_job(app_module.DB_PATH, "g-legacy")["finished_at"] is not None
    assert fetch_job(app_module.DB_PATH, "g-pending")["finished_at"] is None


def test_requeued_job_is_synthesized_from_new_transcription(worker, client, app_module, model):
    insert_job(app_module.DB_PATH, GUID, status="error", transcription="texto viejo.")

    resp = client.post("/sermon", json={"sermon_guid": GUID, "transcription": "texto nuevo."})
    assert resp.status_code == 201
    worker.process_pending_jobs()

    assert model.calls == ["texto nuevo."]
    assert fetch_job(app_module.DB_PATH, GUID)["status"] == "complete"


def test_submit_sermon_errored_guid_is_requeued(client, app_module):
    insert_job(app_module.DB_PATH, GUID, status="error", transcription="vieja")
    conn = sqlite3.connect(app_module.DB_PATH)
    conn.execute("UPDATE sermons SET finished_at = '2026-01-01 00:00:00' WHERE sermon_guid = 'g-err'")
    conn.commit()
    conn.close()

    resp = client.post("/sermon", json={"sermon_guid": GUID, "transcription": "nueva"})

    assert resp.status_code == 201
    job = fetch_job(app_module.DB_PATH, GUID)
    assert job["status"] == "pending"
    assert job["transcription"] == "nueva"
    assert job["finished_at"] is None
    assert job["audio_file"] is None


@pytest.mark.parametrize("status", ["pending", "complete"])
def test_submit_sermon_existing_non_error_guid_conflicts(client, app_module, status):
    insert_job(app_module.DB_PATH, GUID, status=status, transcription="original")

    resp = client.post("/sermon", json={"sermon_guid": GUID, "transcription": "otra"})

    assert resp.status_code == 409
    job = fetch_job(app_module.DB_PATH, GUID)
    assert job["status"] == status
    assert job["transcription"] == "original"


def complete_job(db_path, guid, audio_file):
    conn = sqlite3.connect(db_path)
    conn.execute(
        "UPDATE sermons SET status = 'complete', audio_file = ?, finished_at = '2026-01-01 00:00:00' "
        "WHERE sermon_guid = ?",
        (audio_file, guid),
    )
    conn.commit()
    conn.close()


@pytest.fixture
def finished(app_module, tmp_path):
    """A completed job whose MP3 is on disk; returns the MP3 path."""
    audio = tmp_path / f"{GUID}.mp3"
    audio.write_bytes(b"old audio")
    insert_job(app_module.DB_PATH, GUID, transcription="vieja")
    complete_job(app_module.DB_PATH, GUID, str(audio))
    return audio


def test_submit_sermon_replace_requeues_complete_job(client, app_module, finished):
    resp = client.post("/sermon", json={"sermon_guid": GUID, "transcription": "nueva", "replace": True})

    assert resp.status_code == 201
    job = fetch_job(app_module.DB_PATH, GUID)
    assert job["status"] == "pending"
    assert job["transcription"] == "nueva"
    assert job["audio_file"] is None
    assert job["finished_at"] is None
    # The old file waits for the worker to swap the new one in, but a pending
    # job is never served.
    assert finished.exists()
    assert client.get(f"/download/{GUID}").status_code == 400
    assert "download_url" not in client.get(f"/status/{GUID}").get_json()


def test_submit_sermon_replace_commit_failure_keeps_complete_job(client, app_module, finished, monkeypatch):
    real_get_db = app_module.get_db

    def get_db():
        db = real_get_db()
        return CommitFails(db)

    class CommitFails:
        def __init__(self, db):
            self._db = db

        def __getattr__(self, name):
            return getattr(self._db, name)

        def commit(self):
            raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(app_module, "get_db", get_db)
    resp = client.post("/sermon", json={"sermon_guid": GUID, "transcription": "nueva", "replace": True})

    assert resp.status_code == 500
    job = fetch_job(app_module.DB_PATH, GUID)
    assert job["status"] == "complete"
    assert job["transcription"] == "vieja"
    assert job["audio_file"] == str(finished)
    assert finished.read_bytes() == b"old audio"


@pytest.mark.parametrize("replace", [None, False, "true", 1])
def test_submit_sermon_complete_job_without_replace_true_conflicts(client, app_module, finished, replace):
    payload = {"sermon_guid": GUID, "transcription": "nueva"}
    if replace is not None:
        payload["replace"] = replace

    resp = client.post("/sermon", json=payload)

    assert resp.status_code == 409
    assert finished.exists()
    job = fetch_job(app_module.DB_PATH, GUID)
    assert job["status"] == "complete"
    assert job["transcription"] == "vieja"


def test_submit_sermon_replace_leaves_pending_job_alone(client, app_module):
    insert_job(app_module.DB_PATH, GUID, status="pending", transcription="original")

    resp = client.post("/sermon", json={"sermon_guid": GUID, "transcription": "otra", "replace": True})

    assert resp.status_code == 409
    assert fetch_job(app_module.DB_PATH, GUID)["transcription"] == "original"


@pytest.mark.parametrize("status", ["error", None])
def test_submit_sermon_replace_on_error_or_new_guid_behaves_as_before(client, app_module, status):
    if status:
        insert_job(app_module.DB_PATH, GUID, status=status, transcription="vieja")

    resp = client.post("/sermon", json={"sermon_guid": GUID, "transcription": "nueva", "replace": True})

    assert resp.status_code == 201
    job = fetch_job(app_module.DB_PATH, GUID)
    assert job["status"] == "pending"
    assert job["transcription"] == "nueva"


def test_replaced_job_is_synthesized_from_new_transcription(worker, client, app_module, model):
    insert_job(app_module.DB_PATH, GUID, transcription="texto viejo.")
    worker.process_pending_jobs()
    old = fetch_job(app_module.DB_PATH, GUID)
    assert old["status"] == "complete"

    resp = client.post("/sermon", json={"sermon_guid": GUID, "transcription": "texto nuevo.", "replace": True})
    assert resp.status_code == 201
    worker.process_pending_jobs()

    assert model.calls == ["texto viejo.", "texto nuevo."]
    job = fetch_job(app_module.DB_PATH, GUID)
    assert job["status"] == "complete"
    assert job["audio_file"] == old["audio_file"]
    assert Path(job["audio_file"]).read_bytes() == b"audio"


def test_replaced_job_swaps_new_audio_in_atomically(worker, client, app_module, model, monkeypatch):
    insert_job(app_module.DB_PATH, GUID, transcription="texto viejo.")
    worker.process_pending_jobs()
    target = Path(fetch_job(app_module.DB_PATH, GUID)["audio_file"])
    target.write_bytes(b"old audio")
    client.post("/sermon", json={"sermon_guid": GUID, "transcription": "texto nuevo.", "replace": True})

    seen = []

    def watch_replace(src, dst):
        # While the new MP3 is being written, the old one is still whole at
        # the final path; the new one lands with a single rename.
        seen.append((Path(src).name, Path(dst).name, target.read_bytes(), Path(src).read_bytes()))
        real_replace(src, dst)

    real_replace = worker.os.replace
    monkeypatch.setattr(worker.os, "replace", watch_replace)
    worker.process_pending_jobs()

    assert seen == [(f"{GUID}.mp3.tmp", f"{GUID}.mp3", b"old audio", b"audio")]
    assert target.read_bytes() == b"audio"
    assert sorted(p.name for p in target.parent.iterdir()) == [f"{GUID}.mp3"]


def test_replaced_job_that_fails_removes_old_audio(worker, client, app_module, model):
    insert_job(app_module.DB_PATH, GUID, transcription="texto viejo.")
    worker.process_pending_jobs()
    target = Path(fetch_job(app_module.DB_PATH, GUID)["audio_file"])
    assert target.exists()

    # Whitespace only: synthesize_text raises before any audio is written.
    client.post("/sermon", json={"sermon_guid": GUID, "transcription": "   ", "replace": True})
    worker.process_pending_jobs()

    job = fetch_job(app_module.DB_PATH, GUID)
    assert job["status"] == "error"
    assert not target.exists()


@pytest.mark.parametrize("guid", ["../x", "g-new", "../../etc/passwd", "{" + GUID + "}", "urn:uuid:" + GUID, 42])
def test_submit_sermon_rejects_guid_that_is_not_a_uuid(client, app_module, guid):
    resp = client.post("/sermon", json={"sermon_guid": guid, "transcription": "Hola."})
    assert resp.status_code == 400


def test_submit_sermon_accepts_uppercase_uuid(client, app_module):
    resp = client.post("/sermon", json={"sermon_guid": GUID.upper(), "transcription": "Hola."})
    assert resp.status_code == 201


@pytest.mark.parametrize("body", ["[1, 2]", '"texto"', "null", "no es json"])
def test_submit_sermon_body_not_a_json_object_is_400(client, app_module, body):
    resp = client.post("/sermon", data=body, content_type="application/json")
    assert resp.status_code == 400


def test_submit_sermon_new_guid_created(client, app_module):
    resp = client.post("/sermon", json={"sermon_guid": GUID, "transcription": "Hola."})
    assert resp.status_code == 201
    assert fetch_job(app_module.DB_PATH, GUID)["status"] == "pending"


class WavTTS:
    """Writes a short tone as WAV, which is what coqui's tts_to_file produces."""

    def tts_to_file(self, text, file_path):
        frames = b"".join(
            struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * n / 22050))) for n in range(2205)
        )
        with wave.open(file_path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(22050)
            w.writeframes(frames)


def is_mp3(data):
    return data[:3] == b"ID3" or (data[0] == 0xFF and data[1] & 0xE0 == 0xE0)


needs_real_pydub = pytest.mark.skipif(
    not getattr(sys.modules.get("pydub"), "AudioSegment", None) or not shutil.which("ffmpeg"),
    reason="needs pydub and ffmpeg",
)


@needs_real_pydub
@pytest.mark.parametrize(
    "text",
    ["Hola a todos.", "Hola a todos, hermanos y hermanas. " * 30],
    ids=["single-segment", "multi-segment"],
)
def test_synthesize_text_writes_real_mp3(worker, tmp_path, text):
    out = tmp_path / "out.mp3"
    worker.synthesize_text(WavTTS(), text, str(out))
    data = out.read_bytes()
    assert data[:4] != b"RIFF"
    assert is_mp3(data), data[:4]
    assert [p.name for p in tmp_path.iterdir()] == ["out.mp3"]


# Bitrates (kbps) by header index for MPEG-2 Layer III, which is what LAME
# writes at 22050 Hz.
MPEG2_L3_KBPS = [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160]


def mp3_frames(data):
    """Yield (kbps, sample_rate, mono) for every MPEG-2 Layer III frame."""
    i = 0
    if data[:3] == b"ID3":
        i = 10 + ((data[6] << 21) | (data[7] << 14) | (data[8] << 7) | data[9])
    while i + 4 <= len(data):
        b1, b2, b3 = data[i + 1], data[i + 2], data[i + 3]
        assert data[i] == 0xFF and b1 & 0xFE == 0xF2, f"no MPEG-2 Layer III frame at {i}"
        kbps = MPEG2_L3_KBPS[b2 >> 4]
        sample_rate = [22050, 24000, 16000][(b2 >> 2) & 3]
        yield kbps, sample_rate, b3 >> 6 == 3
        i += 72000 * kbps // sample_rate + ((b2 >> 1) & 1)


@needs_real_pydub
def test_synthesize_text_exports_96k_mono_22050(worker, tmp_path):
    out = tmp_path / "out.mp3"
    worker.synthesize_text(WavTTS(), "Hola a todos, hermanos y hermanas. " * 30, str(out))
    frames = list(mp3_frames(out.read_bytes()))
    assert len(frames) > 1
    assert set(frames) == {(96, 22050, True)}


def test_app_uses_worker_db_and_audio_paths(worker, monkeypatch, tmp_path):
    import importlib
    from unittest import mock

    import app

    monkeypatch.setenv("DB_PATH", str(tmp_path / "custom.db"))
    monkeypatch.setenv("AUDIO_DIR", str(tmp_path / "audio"))
    try:
        importlib.reload(worker)
        with mock.patch("os.makedirs"):
            importlib.reload(app)
        assert app.DB_PATH == str(tmp_path / "custom.db")
        assert app.AUDIO_DIR == str(tmp_path / "audio")
    finally:
        monkeypatch.delenv("DB_PATH")
        monkeypatch.delenv("AUDIO_DIR")
        importlib.reload(worker)
        with mock.patch("os.makedirs"):
            importlib.reload(app)
