from flask import Flask, request, jsonify, send_file, g
import sqlite3
import os
import logging
import threading
from sermon_worker import AUDIO_DIR, DB_PATH, background_worker_loop, purge_old_jobs

# Configure logging with emojis
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure the necessary directories exist. DB_PATH and AUDIO_DIR come from
# sermon_worker so the API and the worker always use the same database.
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)

app = Flask(__name__)

def get_db():
    """Open a new database connection if there is none for the current app context."""
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DB_PATH)
        db.row_factory = sqlite3.Row
    return db

def init_db():
    """Initializes the database with the sermons table."""
    with app.app_context():
        db = get_db()
        cursor = db.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sermons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sermon_guid TEXT NOT NULL UNIQUE,
                transcription TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                audio_file TEXT DEFAULT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                finished_at TIMESTAMP DEFAULT NULL
            )
        ''')
        # Error rows written before the worker recorded finished_at would never
        # match the purge. Stamping them now lets the purge clear them in 24h.
        cursor.execute(
            "UPDATE sermons SET finished_at = CURRENT_TIMESTAMP WHERE status = 'error' AND finished_at IS NULL"
        )
        db.commit()
    logging.info("[APP] INIT_DB - Database initialized successfully.")

@app.teardown_appcontext
def close_db_connection(exception):
    """Closes the database connection at the end of each request."""
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

@app.route('/sermon', methods=['POST'])
def submit_sermon():
    """
    Endpoint for submitting a sermon job.
    Expects a JSON payload with 'sermon_guid' and 'transcription'.
    """
    try:
        data = request.get_json()
        sermon_guid = data.get('sermon_guid')
        transcription = data.get('transcription')
        
        if not sermon_guid or not transcription:
            logging.error("[APP] SUBMIT_SERMON - Missing sermon_guid or transcription in request.")
            return jsonify({"error": "sermon_guid and transcription are required"}), 400
        
        db = get_db()
        cursor = db.cursor()
        # A job that previously failed is requeued in place, so the caller can
        # retry a sermon without waiting for the purge to clear the old row.
        cursor.execute('''
            UPDATE sermons
            SET transcription = ?, status = 'pending', audio_file = NULL,
                created_at = CURRENT_TIMESTAMP, finished_at = NULL
            WHERE sermon_guid = ? AND status = 'error'
        ''', (transcription, sermon_guid))
        if cursor.rowcount:
            db.commit()
            logging.info(f"[APP] SUBMIT_SERMON - Errored sermon job resubmitted: {sermon_guid}")
            return jsonify({"message": "Sermon job submitted successfully"}), 201

        # Check for duplicate sermon_guid
        cursor.execute("SELECT id FROM sermons WHERE sermon_guid = ?", (sermon_guid,))
        if cursor.fetchone():
            logging.warning(f"[APP] SUBMIT_SERMON - Duplicate sermon_guid: {sermon_guid}")
            return jsonify({"error": "A job with this sermon_guid already exists"}), 409
        
        cursor.execute('''
            INSERT INTO sermons (sermon_guid, transcription, status)
            VALUES (?, ?, 'pending')
        ''', (sermon_guid, transcription))
        db.commit()
        logging.info(f"[APP] SUBMIT_SERMON - Sermon job submitted: {sermon_guid}")
        return jsonify({"message": "Sermon job submitted successfully"}), 201
    except Exception as e:
        logging.exception("[APP] SUBMIT_SERMON - Error in submit_sermon endpoint.")
        return jsonify({"error": str(e)}), 500

@app.route('/download/<sermon_guid>', methods=['GET'])
def download_audio(sermon_guid):
    """
    Endpoint to download the completed audio file (mp3) for a given sermon_guid.
    """
    try:
        db = get_db()
        cursor = db.cursor()
        cursor.execute("SELECT audio_file, status FROM sermons WHERE sermon_guid = ?", (sermon_guid,))
        job = cursor.fetchone()
        if not job:
            logging.warning(f"[APP] DOWNLOAD - Sermon job not found: {sermon_guid}")
            return jsonify({"error": "Sermon job not found"}), 404
        if job["status"] != "complete" or not job["audio_file"]:
            logging.warning(f"[APP] DOWNLOAD - Sermon job not complete: {sermon_guid}")
            return jsonify({"error": "Sermon job is not complete yet"}), 400
        
        logging.info(f"[APP] DOWNLOAD - Sending audio file for sermon {sermon_guid}")
        return send_file(job["audio_file"], as_attachment=True)
    except Exception as e:
        logging.exception("[APP] DOWNLOAD - Error in download_audio endpoint.")
        return jsonify({"error": str(e)}), 500

@app.route('/status/<sermon_guid>', methods=['GET'])
def check_status(sermon_guid):
    """
    Endpoint to check the status of a submitted job.
    Returns pending, error, or complete. If complete, includes a download link.
    """
    try:
        db = get_db()
        cursor = db.cursor()
        cursor.execute("SELECT status, audio_file, created_at, finished_at FROM sermons WHERE sermon_guid = ?", (sermon_guid,))
        job = cursor.fetchone()
        if not job:
            logging.warning(f"[APP] DOWNLOAD - Sermon job not found: {sermon_guid}")
            return jsonify({"error": "Sermon job not found"}), 404
        
        response = {
            "sermon_guid": sermon_guid,
            "status": job["status"],
            "created_at": job["created_at"],
            "finished_at": job["finished_at"]
        }
        if job["status"] == "complete" and job["audio_file"]:
            # Assume the server URL is the same host:port
            response["download_url"] = f"/download/{sermon_guid}"
        
        logging.info(f"[APP] STATUS - Status checked for sermon {sermon_guid}: {job['status']}")
        return jsonify(response), 200
    except Exception as e:
        logging.exception("[APP] STATUS - Error in check_status endpoint.")
        return jsonify({"error": str(e)}), 500

@app.route('/')
def list_pending():
    """
    Default endpoint that shows all pending jobs in a simple HTML page.
    The page auto-refreshes every minute.
    """
    try:
        db = get_db()
        cursor = db.cursor()
        cursor.execute("SELECT sermon_guid, created_at FROM sermons WHERE status = 'pending'")
        jobs = cursor.fetchall()
        html = """
        <html>
        <head>
            <meta http-equiv="refresh" content="60">
            <title>Pending Sermon Jobs</title>
            <style>
                table { border-collapse: collapse; width: 50%; margin: 20px auto; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <h2 style="text-align:center;">Pending Sermon Jobs</h2>
            <table>
                <tr>
                    <th>Sermon GUID</th>
                    <th>Created At</th>
                </tr>
        """
        for job in jobs:
            html += f"<tr><td>{job['sermon_guid']}</td><td>{job['created_at']}</td></tr>"
        html += """
            </table>
        </body>
        </html>
        """
        logging.info("[APP] LIST_PENDING - Displayed pending sermon jobs.")
        return html, 200
    except Exception as e:
        logging.exception("[APP] LIST_PENDING - Error in list_pending endpoint.")
        return f"<p>Error: {str(e)}</p>", 500

# (Optional) Endpoint to manually trigger purge (for testing/debugging)
@app.route('/purge', methods=['POST'])
def manual_purge():
    try:
        purge_old_jobs()
        logging.info("[APP] MANUAL_PURGE - Manual purge executed.")
        return jsonify({"message": "Old jobs purged successfully."}), 200
    except Exception as e:
        logging.exception("[APP] MANUAL_PURGE - Error in manual_purge endpoint.")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    logging.info("[APP] STARTUP - Starting Sermon TTS API Server...")
    init_db()

    worker_thread = threading.Thread(target=background_worker_loop, daemon=True)
    worker_thread.start()

    logging.info("[APP] STARTUP - Sermon TTS API Server started successfully on port 5055.")
    app.run(host='0.0.0.0', port=5055, debug=True, use_reloader=False)
