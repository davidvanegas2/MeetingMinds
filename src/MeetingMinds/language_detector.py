from typing import Optional

class LanguageDetector:
    def __init__(self):
        try:
            from langdetect import detect
            self._detect = detect
        except ImportError:
            raise ImportError("Please install langdetect: pip install langdetect")

    def detect_language(self, text: str) -> Optional[str]:
        try:
            lang = self._detect(text)
            return lang
        except Exception:
            return None
