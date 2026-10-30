import sqlite3

import pytest
from helpers import run_on

# Longest text a single tts_to_file() call may receive (text_splitter's
# MAX_SEGMENT_CHARS). Written out so these tests also run against code that
# predates that module.
SAFE_MAX = 400


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
    monkeypatch.setattr(worker, "combine_audio_files", lambda files, out: joined.append(list(files)))
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
        worker.synthesize_text(tts, run_on(2000), str(tmp_path / "out.mp3"))
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


def test_process_pending_jobs_error_sets_finished_at(worker, app_module, monkeypatch, tmp_path):
    insert_job(app_module.DB_PATH, "g-1", transcription="   ")
    monkeypatch.setattr(worker, "DB_PATH", app_module.DB_PATH)
    monkeypatch.setattr(worker, "AUDIO_DIR", str(tmp_path))
    monkeypatch.setattr(worker.TTSModelSingleton, "get_instance", classmethod(lambda cls: FakeTTS()))

    worker.process_pending_jobs()

    job = fetch_job(app_module.DB_PATH, "g-1")
    assert job["status"] == "error"
    assert job["finished_at"] is not None


def test_submit_sermon_errored_guid_is_requeued(client, app_module):
    insert_job(app_module.DB_PATH, "g-err", status="error", transcription="vieja")
    conn = sqlite3.connect(app_module.DB_PATH)
    conn.execute("UPDATE sermons SET finished_at = '2026-01-01 00:00:00' WHERE sermon_guid = 'g-err'")
    conn.commit()
    conn.close()

    resp = client.post("/sermon", json={"sermon_guid": "g-err", "transcription": "nueva"})

    assert resp.status_code == 201
    job = fetch_job(app_module.DB_PATH, "g-err")
    assert job["status"] == "pending"
    assert job["transcription"] == "nueva"
    assert job["finished_at"] is None
    assert job["audio_file"] is None


@pytest.mark.parametrize("status", ["pending", "complete"])
def test_submit_sermon_existing_non_error_guid_conflicts(client, app_module, status):
    insert_job(app_module.DB_PATH, "g-dup", status=status, transcription="original")

    resp = client.post("/sermon", json={"sermon_guid": "g-dup", "transcription": "otra"})

    assert resp.status_code == 409
    job = fetch_job(app_module.DB_PATH, "g-dup")
    assert job["status"] == status
    assert job["transcription"] == "original"


def test_submit_sermon_new_guid_created(client, app_module):
    resp = client.post("/sermon", json={"sermon_guid": "g-new", "transcription": "Hola."})
    assert resp.status_code == 201
    assert fetch_job(app_module.DB_PATH, "g-new")["status"] == "pending"
