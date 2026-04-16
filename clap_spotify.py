"""
Jarvis Audio-to-Spotify
Controla Spotify con aplausos y silbidos.

Comandos:
  2 aplausos rápidos  → play / pause
  3 aplausos rápidos  → siguiente canción
  4 aplausos rápidos  → canción anterior
  Silbido corto       → volumen -10%
  Silbido largo       → volumen +10%

Setup:
  export SPOTIFY_CLIENT_ID=...
  export SPOTIFY_CLIENT_SECRET=...
  export SPOTIFY_REDIRECT_URI=http://127.0.0.1:8888/callback
  python clap_spotify.py
"""

import os
import sys
import time
import logging
import threading
import numpy as np
import sounddevice as sd
import spotipy
from spotipy.oauth2 import SpotifyOAuth

# ── Configuración Spotify ─────────────────────────────────────────────────────

SPOTIFY_CLIENT_ID     = os.getenv("SPOTIFY_CLIENT_ID", "TU_CLIENT_ID_AQUI")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "TU_CLIENT_SECRET_AQUI")
SPOTIFY_REDIRECT_URI  = os.getenv("SPOTIFY_REDIRECT_URI", "http://localhost:8888/callback")
SPOTIFY_TRACK_URI     = os.getenv("SPOTIFY_TRACK_URI", "")

# ── Parámetros de audio ───────────────────────────────────────────────────────

SAMPLE_RATE    = 44100
CHUNK_DURATION = 0.04

# Aplausos
CLAP_MIN_RMS   = 0.10   # energía mínima para contar como aplauso
CLAP_ONSET     = 10.0   # ratio de subida súbita (filtra música)
CLAP_COOLDOWN  = 0.4    # segundos entre aplausos individuales
CLAP_WINDOW    = 2.0    # ventana para contar aplausos

# Silbidos
WHISTLE_FREQ_LOW   = 800    # Hz — límite inferior del rango de silbido
WHISTLE_FREQ_HIGH  = 3000   # Hz — límite superior
WHISTLE_MIN_RMS    = 0.01   # energía mínima para considerar silbido
WHISTLE_DOMINANCE  = 6.0    # pico espectral debe ser X veces la media
WHISTLE_SHORT_MAX  = 0.6    # segundos — por debajo = silbido corto (vol -)
VOLUME_STEP        = 10     # porcentaje de cambio de volumen por silbido

# Cooldown global tras cualquier acción
ACTION_COOLDOWN = 30.0

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("jarvis-spotify")

# ── Acciones Spotify ──────────────────────────────────────────────────────────

def get_active_device(sp: spotipy.Spotify):
    devices = sp.devices()["devices"]
    active = next((d for d in devices if d["is_active"]), None)
    return active or next(iter(devices), None)


def action_play_pause(sp: spotipy.Spotify) -> None:
    state = sp.current_playback()
    if state and state["is_playing"]:
        sp.pause_playback()
        log.info("⏸  Pausado")
    else:
        if SPOTIFY_TRACK_URI:
            device = get_active_device(sp)
            if device:
                sp.start_playback(device_id=device["id"], uris=[SPOTIFY_TRACK_URI])
        else:
            sp.start_playback()
        log.info("▶  Reproduciendo")


def action_next(sp: spotipy.Spotify) -> None:
    sp.next_track()
    log.info("⏭  Siguiente canción")


def action_previous(sp: spotipy.Spotify) -> None:
    sp.previous_track()
    log.info("⏮  Canción anterior")


def action_volume(sp: spotipy.Spotify, direction: str) -> None:
    device = get_active_device(sp)
    if device is None:
        return
    current = device.get("volume_percent", 50) or 50
    new_vol = max(0, min(100, current + (VOLUME_STEP if direction == "up" else -VOLUME_STEP)))
    sp.volume(new_vol, device_id=device["id"])
    arrow = "🔊" if direction == "up" else "🔉"
    log.info("%s  Volumen: %d%%", arrow, new_vol)


# ── Detección ─────────────────────────────────────────────────────────────────

class AudioDetector:
    def __init__(self) -> None:
        self._prev_rms: float = 0.0
        self._last_clap_time: float = 0.0
        self._clap_times: list[float] = []

        self._whistle_frames: int = 0          # frames consecutivos con silbido
        self._was_whistling: bool = False
        self._whistle_start: float = 0.0

        self._action_queue: list[tuple] = []   # (action, args)
        self._lock = threading.Lock()
        self._event = threading.Event()

    # ── Detección de aplauso ──────────────────────────────────────────────────

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
            log.debug("Aplauso #%d (rms=%.3f ratio=%.1f)", len(self._clap_times), rms, rms / prev)

        self._clap_times = [t for t in self._clap_times if now - t <= CLAP_WINDOW]

        count = len(self._clap_times)
        if count >= 4:
            self._clap_times.clear()
            self._queue("previous")
        elif count == 3 and (now - self._last_clap_time) > CLAP_COOLDOWN * 2:
            self._clap_times.clear()
            self._queue("next")
        elif count == 2 and (now - self._last_clap_time) > CLAP_COOLDOWN * 2:
            self._clap_times.clear()
            self._queue("play_pause")

    # ── Detección de silbido ──────────────────────────────────────────────────

    def _check_whistle(self, chunk: np.ndarray, rms: float, now: float) -> None:
        chunk_size = len(chunk)
        fft_mag = np.abs(np.fft.rfft(chunk))
        freqs   = np.fft.rfftfreq(chunk_size, d=1.0 / SAMPLE_RATE)

        mask = (freqs >= WHISTLE_FREQ_LOW) & (freqs <= WHISTLE_FREQ_HIGH)
        if not mask.any():
            return

        band_mag  = fft_mag[mask]
        peak      = float(band_mag.max())
        mean_all  = float(fft_mag.mean()) + 1e-9

        is_whistle = (
            rms >= WHISTLE_MIN_RMS
            and peak / mean_all >= WHISTLE_DOMINANCE
        )

        if is_whistle:
            if not self._was_whistling:
                self._was_whistling = True
                self._whistle_start = now
            self._whistle_frames += 1
        else:
            if self._was_whistling:
                duration = now - self._whistle_start
                log.debug("Silbido terminado duración=%.2fs", duration)
                if duration >= 0.15:   # ignora ruidos muy cortos
                    direction = "up" if duration >= WHISTLE_SHORT_MAX else "down"
                    self._queue("volume", direction)
            self._was_whistling = False
            self._whistle_frames = 0

    # ── Cola de acciones ──────────────────────────────────────────────────────

    def _queue(self, action: str, *args) -> None:
        self._action_queue.append((action, args))
        self._event.set()
        log.debug("Acción encolada: %s %s", action, args)

    def process_chunk(self, chunk: np.ndarray) -> None:
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        now = time.monotonic()
        with self._lock:
            self._check_clap(rms, now)
            self._check_whistle(chunk, rms, now)
            self._prev_rms = rms

    def next_action(self) -> tuple | None:
        self._event.wait()
        self._event.clear()
        with self._lock:
            if self._action_queue:
                return self._action_queue.pop(0)
        return None

    def reset(self) -> None:
        with self._lock:
            self._clap_times.clear()
            self._prev_rms = 0.0
            self._last_clap_time = time.monotonic()
            self._was_whistling = False
            self._whistle_frames = 0
            self._action_queue.clear()
        self._event.clear()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if "TU_CLIENT_ID" in SPOTIFY_CLIENT_ID:
        sys.exit("ERROR: Falta configurar SPOTIFY_CLIENT_ID.")

    log.info("Conectando con Spotify…")
    scope = (
        "user-modify-playback-state "
        "user-read-playback-state "
        "user-read-currently-playing"
    )
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET,
        redirect_uri=SPOTIFY_REDIRECT_URI,
        scope=scope,
        open_browser=True,
    ))
    log.info("Conexión exitosa.")

    detector = AudioDetector()
    chunk_size = int(SAMPLE_RATE * CHUNK_DURATION)

    def audio_callback(indata, frames, time_info, status):
        if status:
            log.warning("Audio: %s", status)
        mono = indata[:, 0] if indata.ndim > 1 else indata.flatten()
        detector.process_chunk(mono)

    log.info("Escuchando:")
    log.info("  2 aplausos → play/pause")
    log.info("  3 aplausos → siguiente")
    log.info("  4 aplausos → anterior")
    log.info("  Silbido corto → volumen -10%%")
    log.info("  Silbido largo → volumen +10%%")
    log.info("  Ctrl+C para salir")

    handlers = {
        "play_pause": lambda args: action_play_pause(sp),
        "next":       lambda args: action_next(sp),
        "previous":   lambda args: action_previous(sp),
        "volume":     lambda args: action_volume(sp, args[0]),
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
                action = detector.next_action()
                if action is None:
                    continue
                name, args = action
                try:
                    handlers[name](args)
                except Exception as exc:
                    log.error("Error en acción '%s': %s", name, exc)
                log.info("Bloqueado %.0fs…", ACTION_COOLDOWN)
                time.sleep(ACTION_COOLDOWN)
                detector.reset()
    except KeyboardInterrupt:
        log.info("Detenido.")
    except sd.PortAudioError as exc:
        sys.exit(f"Error de micrófono: {exc}")


if __name__ == "__main__":
    main()
