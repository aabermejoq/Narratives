"""
Jarvis Cumbia
  - 2 aplausos  → reproduce una cumbia random en Spotify
  - "Buenos días Jarvis" → "Buenos días Aaron" + frase motivacional

Setup:
  export SPOTIFY_CLIENT_ID=...
  export SPOTIFY_CLIENT_SECRET=...
  export SPOTIFY_REDIRECT_URI=http://127.0.0.1:8888/callback
  python clap_spotify.py
"""

import os
import sys
import time
import random
import queue
import logging
import threading
import numpy as np
import sounddevice as sd
import spotipy
import pyttsx3
import speech_recognition as sr
from spotipy.oauth2 import SpotifyOAuth

# ── Configuración ─────────────────────────────────────────────────────────────

SPOTIFY_CLIENT_ID     = os.getenv("SPOTIFY_CLIENT_ID", "TU_CLIENT_ID_AQUI")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "TU_CLIENT_SECRET_AQUI")
SPOTIFY_REDIRECT_URI  = os.getenv("SPOTIFY_REDIRECT_URI", "http://localhost:8888/callback")

SAMPLE_RATE    = 44100
CHUNK_DURATION = 0.04

CLAP_MIN_RMS  = 0.06
CLAP_ONSET    = 6.0
CLAP_COOLDOWN = 0.4
CLAP_WINDOW   = 2.0

PLAY_COOLDOWN = 10.0   # segundos bloqueados tras reproducir

FRASES = [
    "Hoy es un gran día para comerse el mundo.",
    "Cada mañana es una nueva oportunidad.",
    "Tú puedes con todo lo que venga hoy.",
    "Un paso a la vez, Aaron. Tú lo tienes.",
    "Hoy vas a romperla.",
]

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("jarvis")

# ── TTS ───────────────────────────────────────────────────────────────────────

_tts_queue: queue.Queue = queue.Queue()
is_speaking = threading.Event()

def _tts_worker() -> None:
    engine = pyttsx3.init()
    for voice in engine.getProperty("voices"):
        if "spanish" in voice.name.lower() or "es_" in voice.id.lower():
            engine.setProperty("voice", voice.id)
            break
    engine.setProperty("rate", 175)
    while True:
        text = _tts_queue.get()
        if text is None:
            break
        is_speaking.set()
        engine.say(text)
        engine.runAndWait()
        time.sleep(0.5)
        is_speaking.clear()

def speak(text: str) -> None:
    log.info("🔊  %s", text)
    _tts_queue.put(text)

# ── Spotify ───────────────────────────────────────────────────────────────────

def play_random_cumbia(sp: spotipy.Spotify) -> None:
    results = sp.search(q="cumbia", type="track", limit=20, market="MX")
    tracks = results["tracks"]["items"]
    if not tracks:
        log.warning("No se encontraron cumbias.")
        return
    track = random.choice(tracks)
    devices = sp.devices()["devices"]
    device = next((d for d in devices if d["is_active"]), None) or next(iter(devices), None)
    if device is None:
        log.warning("Abre Spotify en tu Mac primero.")
        return
    sp.start_playback(device_id=device["id"], uris=[track["uri"]])
    log.info("▶  %s — %s", track["name"], track["artists"][0]["name"])

# ── Detección de aplausos ─────────────────────────────────────────────────────

class ClapDetector:
    def __init__(self, action_queue: queue.Queue) -> None:
        self._q = action_queue
        self._prev_rms: float = 0.0
        self._last_clap_time: float = 0.0
        self._clap_times: list[float] = []
        self._blocked_until: float = 0.0
        self._lock = threading.Lock()

    def block(self) -> None:
        with self._lock:
            self._blocked_until = time.monotonic() + PLAY_COOLDOWN
            self._clap_times.clear()

    def process_chunk(self, chunk: np.ndarray) -> None:
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        now = time.monotonic()
        with self._lock:
            if now < self._blocked_until:
                self._prev_rms = rms
                return

            prev = self._prev_rms
            self._prev_rms = rms

            is_clap = (
                rms >= CLAP_MIN_RMS
                and prev > 0
                and (rms / prev) >= CLAP_ONSET
                and (now - self._last_clap_time) >= CLAP_COOLDOWN
            )
            if is_clap:
                self._last_clap_time = now
                self._clap_times.append(now)
                log.debug("Aplauso #%d", len(self._clap_times))

            self._clap_times = [t for t in self._clap_times if now - t <= CLAP_WINDOW]

            if (len(self._clap_times) >= 2
                    and (now - self._last_clap_time) > CLAP_COOLDOWN * 2):
                self._clap_times.clear()
                self._q.put("cumbia")

# ── Reconocimiento de voz ─────────────────────────────────────────────────────

_voice_audio_queue: queue.Queue = queue.Queue(maxsize=500)

def voice_listener(action_queue: queue.Queue, stop_event: threading.Event) -> None:
    recognizer = sr.Recognizer()
    chunks_needed = int(4.0 / CHUNK_DURATION)
    log.info("Voz lista.")

    while not stop_event.is_set():
        if is_speaking.is_set():
            while not _voice_audio_queue.empty():
                try:
                    _voice_audio_queue.get_nowait()
                except queue.Empty:
                    break
            time.sleep(0.1)
            continue

        collected: list[np.ndarray] = []
        while len(collected) < chunks_needed and not stop_event.is_set():
            try:
                collected.append(_voice_audio_queue.get(timeout=1))
            except queue.Empty:
                continue

        if is_speaking.is_set() or not collected:
            continue

        try:
            pcm = (np.concatenate(collected) * 32767).astype(np.int16)
            text = recognizer.recognize_google(
                sr.AudioData(pcm.tobytes(), SAMPLE_RATE, 2), language="es-ES"
            ).lower()
            log.info("Voz: '%s'", text)

            if "jarvis" in text and any(w in text for w in ["buenos días", "buenos dias"]):
                action_queue.put("greet")

        except sr.UnknownValueError:
            pass
        except Exception as exc:
            log.error("Error voz: %s", exc)

# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if "TU_CLIENT_ID" in SPOTIFY_CLIENT_ID:
        sys.exit("ERROR: Falta configurar SPOTIFY_CLIENT_ID.")

    log.info("Conectando con Spotify…")
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET,
        redirect_uri=SPOTIFY_REDIRECT_URI,
        scope="user-modify-playback-state user-read-playback-state",
        open_browser=True,
    ))
    log.info("Conexión exitosa.")

    action_queue: queue.Queue = queue.Queue()
    detector = ClapDetector(action_queue)
    stop_event = threading.Event()

    threading.Thread(target=_tts_worker, daemon=True).start()
    threading.Thread(target=voice_listener, args=(action_queue, stop_event), daemon=True).start()

    _start_time = time.monotonic()

    def audio_callback(indata, frames, time_info, status):
        mono = indata[:, 0] if indata.ndim > 1 else indata.flatten()
        if time.monotonic() - _start_time >= 3.0:
            detector.process_chunk(mono)
        if not is_speaking.is_set():
            try:
                _voice_audio_queue.put_nowait(mono.copy())
            except queue.Full:
                pass

    log.info("Listo. Aplaude 2 veces o di 'Buenos días Jarvis'.")

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                            blocksize=int(SAMPLE_RATE * CHUNK_DURATION),
                            callback=audio_callback):
            while True:
                try:
                    action = action_queue.get(timeout=1)
                except queue.Empty:
                    continue

                while not action_queue.empty():
                    try:
                        action_queue.get_nowait()
                    except queue.Empty:
                        break

                if action == "cumbia":
                    log.info("¡Aplauso! Buscando cumbia…")
                    try:
                        play_random_cumbia(sp)
                    except Exception as exc:
                        log.error("%s", exc)
                    detector.block()
                    while is_speaking.is_set():
                        time.sleep(0.1)
                    time.sleep(PLAY_COOLDOWN)

                elif action == "greet":
                    frase = random.choice(FRASES)
                    speak(f"Buenos días Aaron. {frase}")
                    while is_speaking.is_set():
                        time.sleep(0.1)

    except KeyboardInterrupt:
        stop_event.set()
        _tts_queue.put(None)
        log.info("Hasta luego.")
    except sd.PortAudioError as exc:
        sys.exit(f"Error de micrófono: {exc}")


if __name__ == "__main__":
    main()
