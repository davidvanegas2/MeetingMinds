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

# Backend interface
@runtime_checkable
class DiarizationBackend(Protocol):
    def diarize(self, audio_path: Path, transcript_segments: Optional[List] = None) -> DiarizationResult:
        ...

# Pyannote implementation
class PyannoteDiarizationBackend:
    def __init__(self, pipeline_name: str = "pyannote/speaker-diarization-3.1", access_token: Optional[str] = None):
        from pyannote.audio import Pipeline
        self.pipeline = Pipeline.from_pretrained(pipeline_name, use_auth_token=access_token)

    def diarize(self, audio_path: Path) -> DiarizationResult:
        diarization = self.pipeline(str(audio_path))
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(SpeakerSegment(start=turn.start, end=turn.end, speaker=speaker))
        # Optionally align transcript segments to speakers here
        return DiarizationResult(segments=segments)

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    if len(sys.argv) < 2:
        print("Usage: python diarizer.py <audio_file>")
        sys.exit(1)

    # Initialize diarization backend (replace with your HuggingFace token if needed)
    diarizer = PyannoteDiarizationBackend(access_token="")

    # Run diarization
    result = diarizer.diarize(Path(sys.argv[1]))

    # Print results
    for segment in result.segments:
        print(f"Speaker: {segment.speaker}, Start: {segment.start:.2f}s, End: {segment.end:.2f}s")
