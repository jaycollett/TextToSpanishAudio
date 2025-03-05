import os
# Set PyTorch environment variable to help reduce memory fragmentation.
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import sqlite3
import logging
import time
import torch
import gc
from datetime import datetime, timedelta
from TTS.api import TTS

# Configure logging with emojis
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

DB_PATH = "/data/jobs.db"
AUDIO_DIR = "/data/audiofiles"

# Ensure the audio directory exists
os.makedirs(AUDIO_DIR, exist_ok=True)

# Load the TTS model (using the specified Spanish voice model) only once
logging.info("🎤 Loading TTS model...")
tts = TTS("tts_models/es/css10/vits", gpu=True)
logging.info("✅ TTS model loaded.")

# Set the model to evaluation mode and convert to half precision to reduce VRAM usage.
try:
    tts.model.eval()
except Exception as e:
    logging.warning("⚠️ Could not set model to eval: " + str(e))

def process_pending_jobs():
    """
    Check the database for pending sermon jobs.
    For each pending job, generate an MP3 using the preloaded TTS model and update the job status.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT id, sermon_guid, transcription FROM sermons WHERE status = 'pending'")
        jobs = cursor.fetchall()
        for job in jobs:
            sermon_guid = job["sermon_guid"]
            logging.info(f"🎙️ Processing sermon job: {sermon_guid}")
            try:
                output_path = os.path.join(AUDIO_DIR, f"{sermon_guid}.mp3")
                # Wrap inference in no_grad to avoid storing gradients.
                with torch.no_grad():
                    tts.tts_to_file(text=job["transcription"], file_path=output_path)
                # Run garbage collection and clear GPU cache to reduce memory fragmentation.
                gc.collect()
                torch.cuda.empty_cache()
                finished_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                cursor.execute(
                    "UPDATE sermons SET status = 'complete', finished_at = ?, audio_file = ? WHERE id = ?",
                    (finished_at, output_path, job["id"])
                )
                conn.commit()
                logging.info(f"✅ Completed sermon job: {sermon_guid}")
            except Exception as e:
                logging.error(f"❌ Error processing sermon {sermon_guid}: {e}")
                cursor.execute("UPDATE sermons SET status = 'error' WHERE id = ?", (job["id"],))
                conn.commit()
        conn.close()
    except Exception as e:
        logging.error(f"❌ Error accessing database: {e}")

def daily_purge():
    """
    Purge sermon jobs (complete or error) older than 24 hours and remove their associated audio files.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        threshold = (datetime.utcnow() - timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            "SELECT id, sermon_guid, audio_file FROM sermons WHERE (status IN ('complete', 'error')) AND finished_at <= ?",
            (threshold,)
        )
        jobs = cursor.fetchall()
        for job in jobs:
            if job["audio_file"] and os.path.exists(job["audio_file"]):
                os.remove(job["audio_file"])
                logging.info(f"🗑️ Deleted audio file for sermon {job['sermon_guid']}")
            cursor.execute("DELETE FROM sermons WHERE id = ?", (job["id"],))
            logging.info(f"🗑️ Purged job {job['sermon_guid']} from database")
        conn.commit()
        conn.close()
    except Exception as e:
        logging.error(f"❌ Error purging old jobs: {e}")

def background_worker_loop():
    last_purge = time.time()
    purge_interval = 24 * 60 * 60  # 24 hours
    logging.info("⏰ Starting sermon worker loop...")
    while True:
        process_pending_jobs()
        if time.time() - last_purge >= purge_interval:
            daily_purge()
            last_purge = time.time()
        logging.info("⏰ Sleeping for 60 seconds before next check...")
        time.sleep(60)
