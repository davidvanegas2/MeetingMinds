import logging
from dataclasses import dataclass, field
from typing import List, Protocol, runtime_checkable
from pathlib import Path


# Error hierarchy
class TranscriptionError(Exception):
    """Base exception for transcription errors."""


class AudioFormatError(TranscriptionError):
    """Raised when audio format conversion fails."""


class ModelLoadError(TranscriptionError):
    """Raised when model loading fails."""


class BackendError(TranscriptionError):
    """Raised for backend-specific errors."""


# Data models
@dataclass(frozen=True)
class TranscriptSegment:
    start: float  # seconds
    end: float  # seconds
    text: str


@dataclass(frozen=True)
class Transcript:
    segments: List[TranscriptSegment] = field(default_factory=list)
    full_text: str = ""


# Backend interface
@runtime_checkable
class TranscriptionBackend(Protocol):
    def transcribe(self, audio_path: Path) -> Transcript:
        ...


# Whisper backend implementation
class WhisperBackend:
    def __init__(self, model_name: str = "base"):
        self.model_name = model_name
        self._model = None
        self._logger = logging.getLogger(__name__)

    def _lazy_load_model(self):
        if self._model is None:
            try:
                import whisper
                self._logger.info(f"Loading Whisper model '{self.model_name}'...")
                self._model = whisper.load_model(self.model_name)
                self._logger.info("Whisper model loaded.")
            except Exception as e:
                self._logger.error("Failed to load Whisper model.", exc_info=True)
                raise ModelLoadError(str(e)) from e

    def _convert_audio(self, audio_path: Path) -> Path:
        try:
            from pydub import AudioSegment
            self._logger.debug(f"Checking audio format for {audio_path}")
            if audio_path.suffix.lower() != ".wav":
                self._logger.info(f"Converting {audio_path} to WAV format...")
                audio = AudioSegment.from_file(audio_path)
                wav_path = audio_path.with_suffix(".wav")
                audio.export(wav_path, format="wav")
                self._logger.info(f"Audio converted to {wav_path}")
                return wav_path
            return audio_path
        except Exception as e:
            self._logger.error("Audio format conversion failed.", exc_info=True)
            raise AudioFormatError(str(e)) from e

    def transcribe(self, audio_path: Path) -> Transcript:
        self._lazy_load_model()
        wav_path = self._convert_audio(audio_path)
        try:
            self._logger.info(f"Transcribing {wav_path} with Whisper...")
            result = self._model.transcribe(str(wav_path))
            segments = [
                TranscriptSegment(
                    start=seg["start"],
                    end=seg["end"],
                    text=seg["text"].strip()
                )
                for seg in result.get("segments", [])
            ]
            full_text = result.get("text", "").strip()
            self._logger.info("Transcription complete.")
            return Transcript(segments=segments, full_text=full_text)
        except Exception as e:
            self._logger.error("Transcription failed.", exc_info=True)
            raise BackendError(str(e)) from e


# Backend factory
def get_backend(name: str, **kwargs) -> TranscriptionBackend:
    if name.lower() == "whisper":
        return WhisperBackend(**kwargs)
    raise ValueError(f"Unknown backend: {name}")


# Main transcriber component
class Transcriber:
    def __init__(self, backend_name: str = "whisper", **backend_kwargs):
        self._logger = logging.getLogger(__name__)
        self._logger.info(f"Initializing Transcriber with backend '{backend_name}'")
        self.backend: TranscriptionBackend = get_backend(backend_name, **backend_kwargs)

    def transcribe(self, audio_path: str | Path) -> Transcript:
        path = Path(audio_path)
        self._logger.info(f"Starting transcription for {path}")
        return self.backend.transcribe(path)
