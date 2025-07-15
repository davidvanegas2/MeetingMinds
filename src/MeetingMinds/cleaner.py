import re
from typing import List
from .diarizer import DiarizedTranscript, DiarizedSegment


class Cleaner:
    def __init__(self, stopwords: List[str] = None, language: str = "en"):
        self.language = language
        if stopwords:
            self.stopwords = set(stopwords)
        else:
            self.stopwords = set(self._default_stopwords(language))

    def clean_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(
            r"[^\w\sáéíóúüñ]", "", text
        )  # Remove punctuation/symbols, keep Spanish chars
        text = re.sub(r"\s+", " ", text).strip()
        words = text.split()
        words = [w for w in words if w not in self.stopwords]
        return " ".join(words)

    def clean_diarized_transcript(
        self, diarized: DiarizedTranscript
    ) -> DiarizedTranscript:
        cleaned_segments = [
            DiarizedSegment(
                start=seg.start,
                end=seg.end,
                speaker=seg.speaker,
                text=self.clean_text(seg.text),
            )
            for seg in diarized.segments
        ]
        cleaned_full_text = self.clean_text(diarized.full_text)
        return DiarizedTranscript(
            segments=cleaned_segments, full_text=cleaned_full_text
        )

    def _default_stopwords(self, language: str) -> List[str]:
        if language == "es":
            # Spanish stopwords
            return [
                "el",
                "la",
                "los",
                "las",
                "un",
                "una",
                "unos",
                "unas",
                "de",
                "del",
                "a",
                "y",
                "en",
                "que",
                "con",
                "por",
                "para",
                "es",
                "al",
                "lo",
                "como",
                "más",
                "pero",
                "sus",
                "le",
                "ya",
                "o",
                "sí",
                "no",
                "se",
                "ha",
                "me",
                "mi",
                "te",
                "tu",
                "su",
                "yo",
                "él",
                "ella",
                "nos",
                "vosotros",
                "ellos",
                "ellas",
                "este",
                "esta",
                "estos",
                "estas",
                "eso",
                "esa",
                "esos",
                "esas",
                "aquí",
                "allí",
                "muy",
                "también",
                "porque",
                "cuando",
                "donde",
                "desde",
                "hasta",
                "entre",
                "sobre",
                "sin",
                "tras",
                "durante",
                "antes",
                "después",
                "todo",
                "todos",
                "todas",
                "cada",
                "cual",
                "cuales",
                "quien",
                "quienes",
                "cuyo",
                "cuyos",
                "cuyas",
                "qué",
                "cómo",
                "cuándo",
                "cuánto",
                "cuántos",
                "cuántas",
                "dónde",
                "adónde",
                "porqué",
                "para qué",
                "pues",
                "entonces",
                "ahora",
                "bien",
                "mal",
                "aun",
                "aunque",
                "además",
                "incluso",
                "sino",
                "ya",
                "todavía",
                "aún",
                "quizá",
                "quizás",
                "tal vez",
                "según",
                "igual",
                "mismo",
                "propio",
                "tampoco",
                "ningún",
                "ninguna",
                "ninguno",
                "ningunas",
                "ningunos",
            ]
        # English stopwords
        return [
            "the",
            "is",
            "in",
            "at",
            "which",
            "on",
            "and",
            "a",
            "an",
            "of",
            "to",
            "for",
            "with",
            "that",
            "this",
            "it",
            "as",
            "by",
            "from",
            "or",
            "but",
            "be",
            "are",
            "was",
            "were",
            "has",
            "have",
            "had",
            "not",
            "can",
            "will",
            "would",
            "should",
            "could",
        ]
