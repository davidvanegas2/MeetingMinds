import logging
from dataclasses import dataclass, field
from typing import List, Protocol, runtime_checkable, Optional
from pathlib import Path


# Data models
@dataclass(frozen=True)
class SpeakerSegment:
    start: float  # seconds
    end: float  # seconds
    speaker: str
    text: Optional[str] = None  # Optionally attach transcript text


@dataclass(frozen=True)
class DiarizationResult:
    segments: List[SpeakerSegment] = field(default_factory=list)


@dataclass(frozen=True)
class DiarizedSegment:
    start: float
    end: float
    speaker: str
    text: str


@dataclass(frozen=True)
class DiarizedTranscript:
    segments: List[DiarizedSegment] = field(default_factory=list)
    full_text: str = ""


# Backend interface
@runtime_checkable
class DiarizationBackend(Protocol):
    def diarize(
        self, audio_path: Path, transcript_segments: Optional[List] = None
    ) -> DiarizationResult: ...


# Pyannote implementation
class PyannoteDiarizationBackend:
    def __init__(
        self,
        pipeline_name: str = "pyannote/speaker-diarization-3.1",
        access_token: Optional[str] = None,
    ):
        self._logger = logging.getLogger(__name__)
        self._logger.info(f"Initializing Pyannote Diarization Backend: {pipeline_name}")
        from pyannote.audio import Pipeline

        self.pipeline = Pipeline.from_pretrained(
            pipeline_name, use_auth_token=access_token
        )
        self._logger.info("Pyannote Diarization Backend initialized.")

    def diarize(self, audio_path: Path) -> DiarizationResult:
        self._logger.info(f"Diarizing audio file: {audio_path}")
        diarization = self.pipeline(str(audio_path))
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(
                SpeakerSegment(start=turn.start, end=turn.end, speaker=speaker)
            )
        self._logger.info(f"Diarization completed with {len(segments)} segments.")
        return DiarizationResult(segments=segments)


class DiarizedTranscriptBuilder:
    @staticmethod
    def merge(transcript, diarization_result) -> DiarizedTranscript:
        # Align transcript segments to diarization segments by time overlap
        diarized_segments = []
        for speaker_segment in diarization_result.segments:
            # Collect transcript segments that overlap with this speaker segment
            texts = []
            for tseg in transcript.segments:
                # Check for time overlap
                if (
                    tseg.end > speaker_segment.start
                    and tseg.start < speaker_segment.end
                ):
                    # Optionally, trim text to the overlap
                    texts.append(tseg.text)
            if texts:
                diarized_segments.append(
                    DiarizedSegment(
                        start=speaker_segment.start,
                        end=speaker_segment.end,
                        speaker=speaker_segment.speaker,
                        text=" ".join(texts),
                    )
                )
        full_text = transcript.full_text
        return DiarizedTranscript(segments=diarized_segments, full_text=full_text)
