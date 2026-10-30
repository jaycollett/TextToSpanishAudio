import os
import sqlite3
import logging
import torch
import gc
import time
from datetime import datetime, timedelta
from TTS.api import TTS
from pydub import AudioSegment  # Used for concatenating audio segments
from text_splitter import split_text

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
        logging.info("[WORKER] MODEL - GPU model unloaded and cache cleared.")
    except Exception as e:
        logging.warning("[WORKER] MODEL - Failed to unload GPU model: " + str(e))

def combine_audio_files(file_list, output_file):
    """
    Combine a list of audio files into one using pydub.
    """
    combined = None
    for file in file_list:
        segment = AudioSegment.from_file(file)
        segment = segment.set_channels(1)  # Convert to mono
        segment = segment.set_frame_rate(22050)  # Set sample rate to 22050Hz
        if combined is None:
            combined = segment
        else:
            combined += segment

    # Normalize loudness once over the whole sermon, so short segments are not
    # each raised to full volume on their own.
    combined = combined.normalize()
    # 96 kbps mono is plenty for one speaking voice at 22050 Hz. The old
    # "352k" setting was never honoured: at 22050 Hz the MP3 is MPEG-2 Layer
    # III, whose ceiling is 160 kbps, so LAME was writing 160 kbps. On a
    # 3m22s Spanish speech sample, 96k took the file from 4.05 MB to 2.43 MB.
    combined.export(output_file, format="mp3", bitrate="96k")

def synthesize_text(tts, text, output_file):
    """
    Split text into segments no longer than MAX_SEGMENT_CHARS, synthesize each
    segment, and combine the resulting audio files into one output file.
    """
    segments = split_text(text)
    if not segments:
        raise ValueError("Transcription contains no speakable text")

    # tts_to_file always writes WAV, whatever the file name says, so even a
    # single segment goes through combine_audio_files to become a real MP3.
    temp_files = []
    try:
        for i, segment in enumerate(segments):
            temp_file = output_file + f".part{i}.wav"
            temp_files.append(temp_file)
            tts.tts_to_file(text=segment, file_path=temp_file)
            gc.collect()
            torch.cuda.empty_cache()
        combine_audio_files(temp_files, output_file)
    finally:
        # Remove temporary segment files, including any left by a failed segment.
        for f in temp_files:
            if os.path.exists(f):
                os.remove(f)

def purge_old_jobs():
    """Purge completed or error jobs older than 24 hours and remove associated audio files."""
    conn = None
    purge_count = 0
    file_failures = 0
    db_failures = 0
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
                try:
                    os.remove(job["audio_file"])
                    logging.info(f"[WORKER] PURGE - Removed audio file for sermon {job['sermon_guid']}")
                except Exception as file_err:
                    file_failures += 1
                    logging.error(f"[WORKER] PURGE - Failed to remove audio file for sermon {job['sermon_guid']}: {file_err}")
            try:
                cursor.execute("DELETE FROM sermons WHERE id = ?", (job["id"],))
                logging.info(f"[WORKER] PURGE - Purged job {job['sermon_guid']} from database")
                purge_count += 1
            except Exception as db_err:
                db_failures += 1
                logging.error(f"[WORKER] PURGE - Failed to remove DB entry for sermon {job['sermon_guid']}: {db_err}")
        conn.commit()
    except Exception as e:
        logging.error(f"[WORKER] PURGE - Error during purging old jobs: {e}")
    finally:
        if conn is not None:
            conn.close()
        logging.info(f"[WORKER] PURGE - Purge complete. Jobs purged: {purge_count}, file failures: {file_failures}, DB failures: {db_failures}")

def process_pending_jobs():
    """
    Check the database for pending sermon jobs.
    For each pending job, generate an MP3 using the preloaded TTS model.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT id, sermon_guid, transcription FROM sermons WHERE status = 'pending'")
        jobs = cursor.fetchall()
        for job in jobs:
            sermon_guid = job["sermon_guid"]
            logging.info(f"[WORKER] PROCESS - Processing sermon job: {sermon_guid}")
            output_path = os.path.join(AUDIO_DIR, f"{sermon_guid}.mp3")
            try:
                # Use the singleton normal TTS model instance (GPU mode).
                tts = TTSModelSingleton.get_instance()
                synthesize_text(tts, job["transcription"], output_path)
                logging.info(f"[WORKER] PROCESS - Completed sermon job: {sermon_guid}")
            except Exception as e:
                logging.error(f"[WORKER] PROCESS - Error processing sermon {sermon_guid}: {e}")
                # finished_at lets the 24h purge remove error rows too.
                finished_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                cursor.execute(
                    "UPDATE sermons SET status = 'error', finished_at = ? WHERE id = ? AND status = 'pending'",
                    (finished_at, job["id"])
                )
                conn.commit()
                continue

            # Update the job as complete if the processing was successful.
            finished_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute(
                "UPDATE sermons SET status = 'complete', finished_at = ?, audio_file = ? WHERE id = ? AND status = 'pending'",
                (finished_at, output_path, job["id"])
            )
            conn.commit()

            gc.collect()
            torch.cuda.empty_cache()
        conn.close()
    except Exception as e:
        logging.error(f"[WORKER] DB - Error accessing database: {e}")

def background_worker_loop():
    """
    Background loop that continuously processes pending jobs and purges old jobs every 5 minutes.
    """
    last_purge_time = datetime.utcnow() - timedelta(minutes=5)
    while True:
        try:
            process_pending_jobs()
        except Exception as loop_error:
            logging.error("[WORKER] LOOP - Error in background worker loop: " + str(loop_error))
        
        if datetime.utcnow() - last_purge_time >= timedelta(minutes=5):
            logging.info("[WORKER] PURGE - Running scheduled purge of old jobs...")
            purge_old_jobs()
            last_purge_time = datetime.utcnow()
        
        time.sleep(5)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("[WORKER] STARTUP - Starting background worker loop...")
    background_worker_loop()
