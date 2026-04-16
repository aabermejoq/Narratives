"""
Jarvis Audio-to-Spotify
Controla Spotify con voz, aplausos y silbidos.

Comandos de voz (di "Jarvis, ..."):
  buenos días        → "Buenos días Aaron"
  adelante/siguiente → siguiente canción
  atrás/anterior     → canción anterior
  pausa / play       → play / pause
  sube el volumen    → volumen +10%
  baja el volumen    → volumen -10%

Gestos:
  2 aplausos → play/pause
  3 aplausos → siguiente
  4 aplausos → anterior
  Silbido corto (<0.6s) → volumen -10%
  Silbido largo (≥0.6s) → volumen +10%

Setup:
  export SPOTIFY_CLIENT_ID=...
  export SPOTIFY_CLIENT_SECRET=...
  export SPOTIFY_REDIRECT_URI=http://127.0.0.1:8888/callback
  python clap_spotify.py
"""

import os
import sys
import time
import queue
import logging
import threading
import numpy as np
import sounddevice as sd
import spotipy
import pyttsx3
import speech_recognition as sr
from spotipy.oauth2 import SpotifyOAuth

# ── Configuración Spotify ─────────────────────────────────────────────────────

SPOTIFY_CLIENT_ID     = os.getenv("SPOTIFY_CLIENT_ID", "TU_CLIENT_ID_AQUI")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "TU_CLIENT_SECRET_AQUI")
SPOTIFY_REDIRECT_URI  = os.getenv("SPOTIFY_REDIRECT_URI", "http://localhost:8888/callback")
SPOTIFY_TRACK_URI     = os.getenv("SPOTIFY_TRACK_URI", "")
JARVIS_NAME           = os.getenv("JARVIS_NAME", "Aaron")

# ── Parámetros de audio ───────────────────────────────────────────────────────

SAMPLE_RATE    = 44100
CHUNK_DURATION = 0.04

# Aplausos
CLAP_MIN_RMS  = 0.10
CLAP_ONSET    = 10.0
CLAP_COOLDOWN = 0.4
CLAP_WINDOW   = 2.0

# Silbidos (FFT)
WHISTLE_FREQ_LOW  = 1000
WHISTLE_FREQ_HIGH = 2500
WHISTLE_MIN_RMS   = 0.05
WHISTLE_DOMINANCE = 15.0
WHISTLE_MIN_DUR   = 0.4    # duración mínima para contar como silbido real
WHISTLE_SHORT_MAX = 0.9
VOLUME_STEP       = 10

ACTION_COOLDOWN = 30.0

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("jarvis")

# ── TTS ───────────────────────────────────────────────────────────────────────

_tts_queue: queue.Queue = queue.Queue()
is_speaking = threading.Event()   # True mientras Jarvis habla → mic ignorado

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
        time.sleep(0.5)   # margen extra para que el eco se disipe
        is_speaking.clear()

def speak(text: str) -> None:
    log.info("🔊  %s", text)
    _tts_queue.put(text)

# ── Acciones Spotify ──────────────────────────────────────────────────────────

def get_active_device(sp):
    devices = sp.devices()["devices"]
    return next((d for d in devices if d["is_active"]), None) or next(iter(devices), None)


def action_play_pause(sp) -> None:
    state = sp.current_playback()
    if state and state["is_playing"]:
        sp.pause_playback()
        speak("Pausado")
    else:
        if SPOTIFY_TRACK_URI:
            device = get_active_device(sp)
            if device:
                sp.start_playback(device_id=device["id"], uris=[SPOTIFY_TRACK_URI])
        else:
            sp.start_playback()
        speak("Reproduciendo")


def action_next(sp) -> None:
    sp.next_track()
    speak("Siguiente canción")


def action_previous(sp) -> None:
    sp.previous_track()
    speak("Canción anterior")


def action_volume(sp, direction: str) -> None:
    device = get_active_device(sp)
    if device is None:
        return
    current = device.get("volume_percent", 50) or 50
    new_vol = max(0, min(100, current + (VOLUME_STEP if direction == "up" else -VOLUME_STEP)))
    sp.volume(new_vol, device_id=device["id"])
    speak(f"Volumen al {new_vol} por ciento")


# ── Detección de aplausos y silbidos ──────────────────────────────────────────

class AudioDetector:
    def __init__(self, action_queue: queue.Queue) -> None:
        self._q = action_queue
        self._prev_rms: float = 0.0
        self._last_clap_time: float = 0.0
        self._clap_times: list[float] = []
        self._was_whistling: bool = False
        self._whistle_start: float = 0.0
        self._lock = threading.Lock()
        self._blocked_until: float = 0.0

    def _is_blocked(self, now: float) -> bool:
        return now < self._blocked_until

    def block(self) -> None:
        with self._lock:
            self._blocked_until = time.monotonic() + ACTION_COOLDOWN
            self._clap_times.clear()
            self._prev_rms = 0.0
            self._was_whistling = False

    def _check_clap(self, rms: float, now: float) -> None:
        prev = self._prev_rms
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
        count = len(self._clap_times)
        silence = (now - self._last_clap_time) > CLAP_COOLDOWN * 2

        if count >= 4:
            self._clap_times.clear()
            self._q.put("previous")
        elif count == 3 and silence:
            self._clap_times.clear()
            self._q.put("next")
        elif count == 2 and silence:
            self._clap_times.clear()
            self._q.put("play_pause")

    def _check_whistle(self, chunk: np.ndarray, rms: float, now: float) -> None:
        fft_mag = np.abs(np.fft.rfft(chunk))
        freqs   = np.fft.rfftfreq(len(chunk), d=1.0 / SAMPLE_RATE)
        mask    = (freqs >= WHISTLE_FREQ_LOW) & (freqs <= WHISTLE_FREQ_HIGH)
        if not mask.any():
            return
        peak     = float(fft_mag[mask].max())
        mean_all = float(fft_mag.mean()) + 1e-9
        is_whistle = rms >= WHISTLE_MIN_RMS and (peak / mean_all) >= WHISTLE_DOMINANCE

        if is_whistle:
            if not self._was_whistling:
                self._was_whistling = True
                self._whistle_start = now
        else:
            if self._was_whistling:
                duration = now - self._whistle_start
                if duration >= WHISTLE_MIN_DUR:
                    self._q.put("volume_up" if duration >= WHISTLE_SHORT_MAX else "volume_down")
            self._was_whistling = False

    def process_chunk(self, chunk: np.ndarray) -> None:
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        now = time.monotonic()
        with self._lock:
            if not self._is_blocked(now):
                self._check_clap(rms, now)
                self._check_whistle(chunk, rms, now)
            self._prev_rms = rms


# ── Reconocimiento de voz ─────────────────────────────────────────────────────

def voice_listener(action_queue: queue.Queue, stop_event: threading.Event) -> None:
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True

    log.info("Micrófono de voz listo.")

    while not stop_event.is_set():
        # no empieces a escuchar mientras Jarvis habla
        if is_speaking.is_set():
            is_speaking.wait()
            time.sleep(0.3)
            continue
        try:
            with sr.Microphone() as source:
                audio = recognizer.listen(source, timeout=2, phrase_time_limit=5)

            if is_speaking.is_set():   # descarta si Jarvis empezó a hablar durante la escucha
                continue

            text = recognizer.recognize_google(audio, language="es-ES").lower()
            log.info("Voz: '%s'", text)

            if "jarvis" not in text:
                continue

            if any(w in text for w in ["buenos días", "buenos dias"]):
                action_queue.put(f"greet")
            elif any(w in text for w in ["adelante", "siguiente"]):
                action_queue.put("next")
            elif any(w in text for w in ["atrás", "atras", "anterior"]):
                action_queue.put("previous")
            elif any(w in text for w in ["pausa", "pausar", "para"]):
                action_queue.put("pause")
            elif any(w in text for w in ["play", "reproduce", "continúa", "continua"]):
                action_queue.put("resume")
            elif "sube" in text and "volumen" in text:
                action_queue.put("volume_up")
            elif "baja" in text and "volumen" in text:
                action_queue.put("volume_down")

        except sr.WaitTimeoutError:
            pass
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
        scope="user-modify-playback-state user-read-playback-state user-read-currently-playing",
        open_browser=True,
    ))
    log.info("Conexión exitosa.")

    action_queue: queue.Queue = queue.Queue()
    detector = AudioDetector(action_queue)
    chunk_size = int(SAMPLE_RATE * CHUNK_DURATION)
    stop_event = threading.Event()

    threading.Thread(target=_tts_worker, daemon=True).start()
    threading.Thread(target=voice_listener, args=(action_queue, stop_event), daemon=True).start()

    _start_time = time.monotonic()

    def audio_callback(indata, frames, time_info, status):
        if time.monotonic() - _start_time < 3.0:   # ignora los primeros 3s
            return
        if status:
            log.warning("Audio: %s", status)
        mono = indata[:, 0] if indata.ndim > 1 else indata.flatten()
        detector.process_chunk(mono)

    log.info("Jarvis listo. Di 'Jarvis, buenos días' para empezar.")

    handlers = {
        "play_pause":  lambda: action_play_pause(sp),
        "next":        lambda: action_next(sp),
        "previous":    lambda: action_previous(sp),
        "pause":       lambda: (sp.pause_playback(), speak("Pausado")),
        "resume":      lambda: (sp.start_playback(), speak("Reproduciendo")),
        "volume_up":   lambda: action_volume(sp, "up"),
        "volume_down": lambda: action_volume(sp, "down"),
        "greet":       lambda: speak(f"Buenos días {JARVIS_NAME}"),
    }

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=chunk_size,
            callback=audio_callback,
        ):
            while True:
                try:
                    action = action_queue.get(timeout=1)
                except queue.Empty:
                    continue
                # vacía cola para evitar acumulación
                while not action_queue.empty():
                    try:
                        action_queue.get_nowait()
                    except queue.Empty:
                        break

                log.info("Acción: %s", action)
                try:
                    handlers[action]()
                except Exception as exc:
                    log.error("Error en '%s': %s", action, exc)

                # espera a que termine el TTS antes de volver a escuchar
                while is_speaking.is_set():
                    time.sleep(0.1)

                if action == "greet":
                    time.sleep(1)  # pequeña pausa antes de escuchar de nuevo
                else:
                    detector.block()
                    time.sleep(ACTION_COOLDOWN)
    except KeyboardInterrupt:
        stop_event.set()
        _tts_queue.put(None)
        log.info("Detenido.")
    except sd.PortAudioError as exc:
        sys.exit(f"Error de micrófono: {exc}")


if __name__ == "__main__":
    main()
