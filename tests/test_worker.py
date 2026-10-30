import sqlite3
from pathlib import Path

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
def model(worker, app_module, monkeypatch, tmp_path):
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
    insert_job(app_module.DB_PATH, "g-retry", status="error", transcription="texto viejo.")

    resp = client.post("/sermon", json={"sermon_guid": "g-retry", "transcription": "texto nuevo."})
    assert resp.status_code == 201
    worker.process_pending_jobs()

    assert model.calls == ["texto nuevo."]
    assert fetch_job(app_module.DB_PATH, "g-retry")["status"] == "complete"


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
