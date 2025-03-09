import os
import sqlite3
import logging
import torch
import gc
import time
from datetime import datetime, timedelta
from TTS.api import TTS

# Global configuration values.
DB_PATH = os.getenv("DB_PATH", "/data/jobs.db")
AUDIO_DIR = os.getenv("AUDIO_DIR", "/data/audiofiles")
NORMAL_MODEL_ID = os.getenv("NORMAL_MODEL_ID", "tts_models/es/css10/vits")

class TTSModelSingleton:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = TTS(NORMAL_MODEL_ID, gpu=True)
        return cls._instance

    @classmethod
    def unload_instance(cls):
        cls._instance = None

def unload_normal_model():
    """
    Unload the currently loaded GPU model and clear GPU memory.
    """
    try:
        TTSModelSingleton.unload_instance()
        torch.cuda.empty_cache()
        logging.info("GPU model unloaded and cache cleared.")
    except Exception as e:
        logging.warning("Failed to unload GPU model: " + str(e))

def purge_old_jobs():
    """
    Purge completed or error jobs older than 24 hours and remove associated audio files.
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
                logging.info(f"🗑️ Removed audio file for sermon {job['sermon_guid']}")
            cursor.execute("DELETE FROM sermons WHERE id = ?", (job["id"],))
            logging.info(f"🗑️ Purged job {job['sermon_guid']} from database")
        conn.commit()
        conn.close()
    except Exception as e:
        logging.error(f"❌ Error during purging old jobs: {e}")

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
                        # Wait before reinitializing the GPU model.
                        logging.info("⏳ Waiting 10 seconds before reinitializing GPU model...")
                        time.sleep(10)
                        try:
                            unload_normal_model()
                            _ = TTSModelSingleton.get_instance()
                        except Exception as reinit_error:
                            logging.error("❌ Failed to reinitialize GPU model: " + str(reinit_error))
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

def background_worker_loop():
    """
    Background loop that continuously processes pending jobs and purges old jobs every 5 minutes.
    """
    last_purge_time = datetime.utcnow() - timedelta(minutes=5)
    while True:
        try:
            process_pending_jobs()
        except Exception as loop_error:
            logging.error("❌ Error in background worker loop: " + str(loop_error))
        
        # Run purge every 5 minutes.
        if datetime.utcnow() - last_purge_time >= timedelta(minutes=5):
            logging.info("🧹 Running scheduled purge of old jobs...")
            purge_old_jobs()
            last_purge_time = datetime.utcnow()
        
        # Sleep briefly between iterations to avoid a tight loop.
        time.sleep(5)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("🔥 Starting background worker loop...")
    background_worker_loop()
