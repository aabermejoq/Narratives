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

# URI de la canción → botón derecho en Spotify → Compartir → Copiar URI del tema
SPOTIFY_TRACK_URI = os.getenv(
    "SPOTIFY_TRACK_URI",
    "spotify:track:4uLU6hMCjMI75M1A2tKUQC"   # Never Gonna Give You Up - Rick Astley (ejemplo)
)

# Detección de aplausos
SAMPLE_RATE       = 44100   # Hz
CHUNK_DURATION    = 0.05    # segundos por bloque de audio
CLAP_THRESHOLD    = 0.05    # amplitud RMS mínima (0.0–1.0); bájala si tu micro es poco sensible
CLAP_COOLDOWN     = 0.3     # segundos de espera entre aplausos para evitar dobles disparos
CLAPS_REQUIRED    = 2       # número de aplausos consecutivos para disparar Spotify
CLAP_WINDOW       = 1.5     # ventana de tiempo (s) en la que deben ocurrir los aplausos
PLAY_COOLDOWN     = 5.0     # segundos de bloqueo tras reproducir para evitar disparos múltiples

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
    active = next(
        (d for d in devices["devices"] if d["is_active"]),
        None,
    )
    if active is None:
        active = next(iter(devices["devices"]), None)

    if active is None:
        log.warning("No se encontró ningún dispositivo Spotify activo. Abre Spotify en tu PC/móvil.")
        return

    device_id = active["id"]
    sp.start_playback(device_id=device_id, uris=[track_uri])
    log.info("▶  Reproduciendo %s en '%s'", track_uri, active["name"])


# ── Detección de aplausos ─────────────────────────────────────────────────────

class ClapDetector:
    def __init__(self) -> None:
        self._last_clap_time: float = 0.0
        self._last_play_time: float = 0.0
        self._clap_times: list[float] = []
        self._lock = threading.Lock()

    def process_chunk(self, chunk: np.ndarray) -> bool:
        """Devuelve True cuando se detecta la secuencia de aplausos requerida."""
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        now = time.monotonic()

        with self._lock:
            if rms >= CLAP_THRESHOLD and (now - self._last_clap_time) >= CLAP_COOLDOWN:
                self._last_clap_time = now
                self._clap_times.append(now)
                log.debug("Aplauso detectado (RMS=%.4f)", rms)

            self._clap_times = [t for t in self._clap_times if now - t <= CLAP_WINDOW]

            if len(self._clap_times) >= CLAPS_REQUIRED:
                self._clap_times.clear()
                if (now - self._last_play_time) >= PLAY_COOLDOWN:
                    self._last_play_time = now
                    return True

        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # Validación básica de credenciales
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

    log.info(
        "Escuchando… aplaudE %d vez/veces en %.1fs para reproducir la canción. (Ctrl+C para salir)",
        CLAPS_REQUIRED,
        CLAP_WINDOW,
    )

    def audio_callback(indata: np.ndarray, frames: int, time_info, status) -> None:  # noqa: ARG001
        if status:
            log.warning("Audio status: %s", status)
        mono = indata[:, 0] if indata.ndim > 1 else indata.flatten()
        if detector.process_chunk(mono):
            log.info("¡Aplauso detectado! Lanzando Spotify…")
            try:
                play_track(sp, SPOTIFY_TRACK_URI)
            except Exception as exc:
                log.error("Error al reproducir: %s", exc)

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=chunk_size,
            callback=audio_callback,
        ):
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        log.info("Detenido por el usuario.")
    except sd.PortAudioError as exc:
        sys.exit(f"Error de micrófono: {exc}\n¿Tienes un micrófono conectado?")


if __name__ == "__main__":
    main()
