import logging
from pathlib import Path
from typing import Optional

from .transcriber import Transcriber, Transcript, TranscriptionError
from .diarizer import (
    PyannoteDiarizationBackend,
    DiarizationResult,
    DiarizedTranscriptBuilder,
    DiarizedTranscript,
)
from .language_detector import LanguageDetector
from .cleaner import Cleaner
# from .summarizer import Summarizer, Summary  # Uncomment when summarizer is implemented


class MeetingPipeline:
    def __init__(
        self,
        transcriber: Optional[Transcriber] = None,
        diarizer: Optional[PyannoteDiarizationBackend] = None,
        # summarizer: Optional[Summarizer] = None,  # Uncomment when available
    ):
        self.logger = logging.getLogger(__name__)
        self.transcriber = transcriber or Transcriber(
            backend_name="whisper", model_name="base"
        )
        self.diarizer = diarizer or PyannoteDiarizationBackend(
            access_token=""
        )
        # self.summarizer = summarizer  # Uncomment when available

    def run(self, audio_path: Path):
        self.logger.info(f"Starting pipeline for {audio_path}")
        # 1. Transcription
        try:
            transcript: Transcript = self.transcriber.transcribe(audio_path)
            self.logger.info("Transcription completed.")
        except TranscriptionError as e:
            self.logger.error(f"Transcription failed: {e}")
            raise
        # 2. Diarization
        diarization_result: DiarizationResult = self.diarizer.diarize(audio_path)
        self.logger.info("Diarization completed.")
        # 3. Merge transcript and diarization
        diarized_transcript: DiarizedTranscript = DiarizedTranscriptBuilder.merge(
            transcript, diarization_result
        )
        self.logger.info("Diarized transcript built.")
        # 4. Detect language
        language_detector = LanguageDetector()
        detected_language = language_detector.detect_language(
            diarized_transcript.full_text
        )
        self.logger.info(f"Detected language: {detected_language}")
        # 5. Clean transcript
        cleaner = Cleaner(
            language=detected_language if detected_language in ["en", "es"] else "en"
        )
        cleaned_diarized_transcript = cleaner.clean_diarized_transcript(
            diarized_transcript
        )
        self.logger.info("Cleaned diarized transcript.")
        # 6. Summarization (optional, placeholder)
        # summary: Optional[Summary] = None
        # if self.summarizer:
        #     summary = self.summarizer.summarize(cleaned_diarized_transcript)
        #     self.logger.info("Summarization completed.")
        # Return all results
        return {
            "transcript": transcript,
            "diarization": diarization_result,
            "diarized_transcript": diarized_transcript,
            "detected_language": detected_language,
            "cleaned_diarized_transcript": cleaned_diarized_transcript,
            # "summary": summary,
        }


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <audio_file>")
        sys.exit(1)
    audio_file = Path(sys.argv[1])
    pipeline = MeetingPipeline()
    results = pipeline.run(audio_file)
    print("\nCleaned Diarized Segments:")
    for seg in results["cleaned_diarized_transcript"].segments:
        print(f"[{seg.start:.2f}-{seg.end:.2f}] Speaker: {seg.speaker} | {seg.text}")
    print("\nDetected language:", results["detected_language"])
    # Uncomment below when summarizer is available
    # if results.get("summary"):
    #     print("\nSummary:\n", results["summary"].text)
