import sqlite3
import types

from test_worker import GUID, complete_job, fetch_job, insert_job


class _Cursor:
    """A cursor that runs a hook right after the first fetchall()."""

    def __init__(self, cursor, after_fetchall):
        self._cursor = cursor
        self._after_fetchall = after_fetchall

    def fetchall(self):
        rows = self._cursor.fetchall()
        if self._after_fetchall:
            hook, self._after_fetchall = self._after_fetchall, None
            hook()
        return rows

    def __getattr__(self, name):
        return getattr(self._cursor, name)


class _Connection:
    def __init__(self, conn, after_fetchall):
        object.__setattr__(self, "_conn", conn)
        object.__setattr__(self, "_after_fetchall", after_fetchall)

    def cursor(self):
        return _Cursor(self._conn.cursor(), self._after_fetchall)

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def __setattr__(self, name, value):
        setattr(self._conn, name, value)


def test_purge_keeps_job_replaced_between_select_and_delete(worker, client, app_module, tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "DB_PATH", app_module.DB_PATH)
    audio = tmp_path / f"{GUID}.mp3"
    audio.write_bytes(b"old audio")
    insert_job(app_module.DB_PATH, GUID, transcription="vieja")
    complete_job(app_module.DB_PATH, GUID, str(audio))  # finished long enough ago to purge

    responses = []

    def replace():
        # The replace lands after the purge has selected the row, before it
        # deletes it.
        responses.append(
            client.post("/sermon", json={"sermon_guid": GUID, "transcription": "nueva", "replace": True})
        )

    fake_sqlite3 = types.SimpleNamespace(
        connect=lambda *a, **kw: _Connection(sqlite3.connect(*a, **kw), replace),
        Row=sqlite3.Row,
    )
    monkeypatch.setattr(worker, "sqlite3", fake_sqlite3)
    worker.purge_old_jobs()

    assert [r.status_code for r in responses] == [201]
    job = fetch_job(app_module.DB_PATH, GUID)
    assert job is not None, "the requeued job was purged"
    assert job["status"] == "pending"
    assert job["transcription"] == "nueva"
    assert client.get(f"/status/{GUID}").status_code == 200
    # Not purged, so its file is left for the worker to replace.
    assert audio.exists()


def test_replace_while_database_is_busy_keeps_complete_job_and_file(client, app_module, tmp_path):
    audio = tmp_path / f"{GUID}.mp3"
    audio.write_bytes(b"old audio")
    insert_job(app_module.DB_PATH, GUID, transcription="vieja")
    complete_job(app_module.DB_PATH, GUID, str(audio))

    # An open read transaction keeps the replace from committing; sqlite3
    # gives up after its 5 second busy timeout.
    reader = sqlite3.connect(app_module.DB_PATH, isolation_level=None)
    reader.execute("BEGIN")
    reader.execute("SELECT * FROM sermons").fetchall()
    try:
        resp = client.post("/sermon", json={"sermon_guid": GUID, "transcription": "nueva", "replace": True})
    finally:
        reader.execute("COMMIT")
        reader.close()

    assert resp.status_code == 500
    job = fetch_job(app_module.DB_PATH, GUID)
    assert job["status"] == "complete"
    assert job["audio_file"] == str(audio)
    assert audio.read_bytes() == b"old audio"
    assert client.get(f"/download/{GUID}").status_code == 200
