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

# Define the model IDs
NORMAL_MODEL_ID = "tts_models/es/css10/vits"

# Singleton class for the normal TTS model
class TTSModelSingleton:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            logging.info("🎤 Loading normal TTS model...")
            cls._instance = TTS(NORMAL_MODEL_ID, gpu=True)
            try:
                cls._instance.model.eval()
            except Exception as e:
                logging.warning("⚠️ Could not set normal model to eval: " + str(e))
            logging.info("✅ Normal TTS model loaded and set to eval.")
        return cls._instance

def unload_normal_model():
    """Unload the normal model and clear GPU caches."""
    TTSModelSingleton._instance = None
    gc.collect()
    torch.cuda.empty_cache()


def process_pending_jobs():
    """
    Check the database for pending sermon jobs.
    For each pending job, generate an MP3 using the preloaded TTS model.
    If a CUDA out-of-memory error occurs, try using CPU for inference.
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
            output_path = os.path.join(AUDIO_DIR, f"{sermon_guid}.mp3")
            try:
                # Use the singleton normal TTS model instance (GPU mode).
                tts = TTSModelSingleton.get_instance()
                with torch.no_grad():
                    tts.tts_to_file(text=job["transcription"], file_path=output_path)
                logging.info(f"✅ Completed sermon job with normal (GPU) model: {sermon_guid}")

            except Exception as e:
                # Check if the error is due to CUDA running out of memory.
                if "out of memory" in str(e).lower():
                    logging.warning("⚠️ CUDA OOM detected. Switching to CPU fallback...")
                    # Unload the normal GPU model and clear cache.
                    unload_normal_model()
                    
                    try:
                        # Load the same model on CPU as a fallback.
                        logging.info("🎤 Loading TTS model on CPU for fallback...")
                        cpu_tts = TTS(NORMAL_MODEL_ID, gpu=False)
                        try:
                            cpu_tts.model.eval()
                        except Exception as ex:
                            logging.warning("⚠️ Could not set CPU model to eval: " + str(ex))
                        with torch.no_grad():
                            cpu_tts.tts_to_file(text=job["transcription"], file_path=output_path)
                        logging.info(f"✅ Successfully processed sermon job with CPU fallback: {sermon_guid}")
                    except Exception as cpu_error:
                        logging.error("❌ CPU fallback also failed for sermon " + sermon_guid + ": " + str(cpu_error))
                        cursor.execute("UPDATE sermons SET status = 'error' WHERE id = ?", (job["id"],))
                        conn.commit()
                        continue  # Skip to the next job
                    finally:
                        # Clean up the CPU model and GPU memory.
                        try:
                            del cpu_tts
                        except Exception:
                            pass
                        gc.collect()
                        torch.cuda.empty_cache()
                        # Reinitialize normal GPU model for subsequent requests.
                        unload_normal_model()
                        _ = TTSModelSingleton.get_instance()
                else:
                    logging.error(f"❌ Error processing sermon {sermon_guid}: {e}")
                    cursor.execute("UPDATE sermons SET status = 'error' WHERE id = ?", (job["id"],))
                    conn.commit()
                    continue

            # Update the job as complete if the processing was successful.
            finished_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute(
                "UPDATE sermons SET status = 'complete', finished_at = ?, audio_file = ? WHERE id = ?",
                (finished_at, output_path, job["id"])
            )
            conn.commit()

            # Run garbage collection and clear GPU cache after each job.
            gc.collect()
            torch.cuda.empty_cache()
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
