from langchain_aws import BedrockLLM


class Summarizer:
    def __init__(
        self,
        model_id: str = "anthropic.claude-v2",  # Default Bedrock model
    ):
        self.llm = BedrockLLM(model=model_id)

    def summarize(self, text: str, language: str = "en") -> str:
        prompt = self._build_prompt(text, language)
        return self.llm(prompt)

    def _build_prompt(self, text: str, language: str) -> str:
        if language == "es":
            instructions = "Resume la siguiente transcripción de una reunión, manteniendo los puntos clave y quién dijo qué."
        else:
            instructions = "Summarize the following meeting transcript, keeping key points and who said what."
        prompt = f"{instructions}\n\nTranscript:\n{text}\n\nSummary:"
        return prompt
