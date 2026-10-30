# TextToSpanishAudio
Simple micro-service that uses TTS to generate Spanish audio from Spanish text.

## Data

The service keeps its state in `/data` inside the container:

- `/data/jobs.db` is the SQLite job database.
- `/data/audiofiles/` holds the generated MP3s until the 24 hour purge removes them.

Mount a named volume or a host directory at `/data`, for example
`-v texttospanishaudio-data:/data`. Without a mount, Docker puts `/data` in an
anonymous volume, and a container recreated from a new image starts with an
empty job database. The directory must be writable by UID 1000, the user the
container runs as.
