"""
Jarvis Clap-to-Spotify
Detecta aplausos por el micrófono y reproduce una canción en Spotify.

Setup:
  1. Crea una app en https://developer.spotify.com/dashboard
  2. Copia CLIENT_ID, CLIENT_SECRET y añade http://localhost:8888/callback como Redirect URI
  3. Pon el SPOTIFY_TRACK_URI de la canción que quieras (botón derecho → Compartir → URI)
  4. Exporta las variables de entorno o edítalas directamente abajo
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

# ── Configuración ─────────────────────────────────────────────────────────────

SPOTIFY_CLIENT_ID     = os.getenv("SPOTIFY_CLIENT_ID", "TU_CLIENT_ID_AQUI")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "TU_CLIENT_SECRET_AQUI")
SPOTIFY_REDIRECT_URI  = os.getenv("SPOTIFY_REDIRECT_URI", "http://localhost:8888/callback")

SPOTIFY_TRACK_URI = os.getenv(
    "SPOTIFY_TRACK_URI",
    "spotify:track:4uLU6hMCjMI75M1A2tKUQC"
)

# Detección — ajusta ONSET_RATIO si hace falta
SAMPLE_RATE    = 44100
CHUNK_DURATION = 0.04    # segundos por bloque
MIN_RMS        = 0.10    # energía mínima — aplaudir cerca del micro sube esto fácil
ONSET_RATIO    = 10.0    # ratio de subida súbita — música no llega a esto
CLAP_COOLDOWN  = 0.4     # segundos entre aplausos individuales
CLAPS_REQUIRED = 2       # aplausos para disparar
CLAP_WINDOW    = 1.5     # ventana en segundos
PLAY_COOLDOWN  = 30.0    # segundos bloqueados tras reproducir

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("clap-spotify")

# ── Spotify ───────────────────────────────────────────────────────────────────

def build_spotify_client() -> spotipy.Spotify:
    scope = "user-modify-playback-state user-read-playback-state"
    auth = SpotifyOAuth(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET,
        redirect_uri=SPOTIFY_REDIRECT_URI,
        scope=scope,
        open_browser=True,
    )
    return spotipy.Spotify(auth_manager=auth)


def play_track(sp: spotipy.Spotify, track_uri: str) -> None:
    devices = sp.devices()
    active = next((d for d in devices["devices"] if d["is_active"]), None)
    if active is None:
        active = next(iter(devices["devices"]), None)
    if active is None:
        log.warning("No se encontró ningún dispositivo Spotify activo.")
        return
    sp.start_playback(device_id=active["id"], uris=[track_uri])
    log.info("▶  Reproduciendo en '%s'", active["name"])


# ── Detección de aplausos por onset ──────────────────────────────────────────

class ClapDetector:
    def __init__(self) -> None:
        self._prev_rms: float = 0.0
        self._last_clap_time: float = 0.0
        self._clap_times: list[float] = []
        self._lock = threading.Lock()
        self._triggered = threading.Event()

    def process_chunk(self, chunk: np.ndarray) -> None:
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        now = time.monotonic()

        with self._lock:
            prev = self._prev_rms
            self._prev_rms = rms

            # onset: pico súbito de energía por encima del umbral mínimo
            is_clap = (
                rms >= MIN_RMS
                and prev > 0
                and (rms / prev) >= ONSET_RATIO
                and (now - self._last_clap_time) >= CLAP_COOLDOWN
            )

            if is_clap:
                self._last_clap_time = now
                self._clap_times.append(now)
                log.debug("Aplauso onset rms=%.4f ratio=%.1f total=%d", rms, rms/prev, len(self._clap_times))

            self._clap_times = [t for t in self._clap_times if now - t <= CLAP_WINDOW]

            if len(self._clap_times) >= CLAPS_REQUIRED and not self._triggered.is_set():
                self._clap_times.clear()
                self._triggered.set()

    def wait_for_clap(self) -> None:
        self._triggered.wait()
        self._triggered.clear()

    def reset(self) -> None:
        with self._lock:
            self._clap_times.clear()
            self._prev_rms = 0.0
            self._last_clap_time = time.monotonic()
        self._triggered.clear()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if "TU_CLIENT_ID" in SPOTIFY_CLIENT_ID:
        sys.exit(
            "ERROR: Falta configurar SPOTIFY_CLIENT_ID.\n"
            "Exporta la variable de entorno o edita clap_spotify.py directamente."
        )

    log.info("Conectando con Spotify…")
    sp = build_spotify_client()
    log.info("Conexión exitosa.")

    detector = ClapDetector()
    chunk_size = int(SAMPLE_RATE * CHUNK_DURATION)

    def audio_callback(indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            log.warning("Audio status: %s", status)
        mono = indata[:, 0] if indata.ndim > 1 else indata.flatten()
        detector.process_chunk(mono)

    log.info(
        "Escuchando… aplaude %d veces para reproducir. (Ctrl+C para salir)",
        CLAPS_REQUIRED,
    )

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=chunk_size,
            callback=audio_callback,
        ):
            while True:
                detector.wait_for_clap()
                log.info("¡Aplauso detectado! Lanzando Spotify…")
                try:
                    play_track(sp, SPOTIFY_TRACK_URI)
                except Exception as exc:
                    log.error("Error al reproducir: %s", exc)
                log.info("Bloqueado %.0fs…", PLAY_COOLDOWN)
                time.sleep(PLAY_COOLDOWN)
                detector.reset()
    except KeyboardInterrupt:
        log.info("Detenido por el usuario.")
    except sd.PortAudioError as exc:
        sys.exit(f"Error de micrófono: {exc}\n¿Tienes un micrófono conectado?")


if __name__ == "__main__":
    main()
