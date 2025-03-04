from flask import Flask, request, jsonify, send_file, g
import sqlite3
import os
import logging
import threading
from datetime import datetime, timedelta
from sermon_worker import background_worker_loop

# Configure logging with emojis
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Define directories and database path
DATA_DIR = "/data"
DB_PATH = os.path.join(DATA_DIR, "jobs.db")
AUDIO_DIR = os.path.join(DATA_DIR, "audiofiles")

# Ensure the necessary directories exist
os.makedirs(DATA_DIR, exist_ok=True)
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
        db.commit()
    logging.info("✅ Database initialized successfully.")

def purge_old_jobs():
    """Purge completed or error jobs older than 24 hours and remove associated audio files."""
    try:
        db = get_db()
        cursor = db.cursor()
        threshold = (datetime.utcnow() - timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            "SELECT id, sermon_guid, audio_file FROM sermons WHERE (status IN ('complete', 'error')) AND finished_at <= ?",
            (threshold,)
        )
        jobs = cursor.fetchall()
        for job in jobs:
            if job["audio_file"] and os.path.exists(job["audio_file"]):
                os.remove(job["audio_file"])
                logging.info(f"🗑️ Removed audio file for sermon {job['sermon_guid']}")
            cursor.execute("DELETE FROM sermons WHERE id = ?", (job["id"],))
            logging.info(f"🗑️ Purged job {job['sermon_guid']} from database")
        db.commit()
    except Exception as e:
        logging.error(f"❌ Error during purging old jobs: {e}")

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
            logging.error("⚠️ Missing sermon_guid or transcription in request.")
            return jsonify({"error": "sermon_guid and transcription are required"}), 400
        
        db = get_db()
        cursor = db.cursor()
        # Check for duplicate sermon_guid
        cursor.execute("SELECT id FROM sermons WHERE sermon_guid = ?", (sermon_guid,))
        if cursor.fetchone():
            logging.warning(f"🚫 Duplicate sermon_guid: {sermon_guid}")
            return jsonify({"error": "A job with this sermon_guid already exists"}), 409
        
        cursor.execute('''
            INSERT INTO sermons (sermon_guid, transcription, status)
            VALUES (?, ?, 'pending')
        ''', (sermon_guid, transcription))
        db.commit()
        logging.info(f"📥 Sermon job submitted: {sermon_guid}")
        return jsonify({"message": "Sermon job submitted successfully"}), 201
    except Exception as e:
        logging.exception("❌ Error in submit_sermon endpoint.")
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
            logging.warning(f"🔍 Sermon job not found: {sermon_guid}")
            return jsonify({"error": "Sermon job not found"}), 404
        if job["status"] != "complete" or not job["audio_file"]:
            logging.warning(f"⌛ Sermon job not complete: {sermon_guid}")
            return jsonify({"error": "Sermon job is not complete yet"}), 400
        
        logging.info(f"📤 Sending audio file for sermon {sermon_guid}")
        return send_file(job["audio_file"], as_attachment=True)
    except Exception as e:
        logging.exception("❌ Error in download_audio endpoint.")
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
            logging.warning(f"🔍 Sermon job not found: {sermon_guid}")
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
        
        logging.info(f"📊 Status checked for sermon {sermon_guid}: {job['status']}")
        return jsonify(response), 200
    except Exception as e:
        logging.exception("❌ Error in check_status endpoint.")
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
        logging.info("📋 Displayed pending sermon jobs.")
        return html, 200
    except Exception as e:
        logging.exception("❌ Error in list_pending endpoint.")
        return f"<p>Error: {str(e)}</p>", 500

# (Optional) Endpoint to manually trigger purge (for testing/debugging)
@app.route('/purge', methods=['POST'])
def manual_purge():
    try:
        purge_old_jobs()
        logging.info("🗑️ Manual purge executed.")
        return jsonify({"message": "Old jobs purged successfully."}), 200
    except Exception as e:
        logging.exception("❌ Error in manual_purge endpoint.")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    logging.info("🔥 Starting Sermon TTS API Server...")
    init_db()

    worker_thread = threading.Thread(target=background_worker_loop, daemon=True)
    worker_thread.start()

    logging.info("✅ Sermon TTS API Server started successfully on port 5055.")
    app.run(host='0.0.0.0', port=5055, debug=True, use_reloader=False)
