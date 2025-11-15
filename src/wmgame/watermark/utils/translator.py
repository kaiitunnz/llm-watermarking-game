import os
from abc import ABC, abstractmethod

import backoff
import langcodes
import openai
from dotenv import load_dotenv
from easynmt import EasyNMT


class Translator(ABC):
    @abstractmethod
    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        pass


class OpenAITranslator(Translator):
    def __init__(self, base_url: str, model: str, api_keys: list[str]):
        self.model = model
        self._clients = [
            openai.OpenAI(api_key=api_key, base_url=base_url) for api_key in api_keys
        ]
        self._idx = 0

    def _get_next_client(self) -> openai.OpenAI:
        client = self._clients[self._idx]
        self._idx = (self._idx + 1) % len(self._clients)
        return client

    @backoff.on_exception(
        backoff.constant, (Exception, ValueError), max_tries=5, interval=2, jitter=None
    )
    def _make_request(self, user_prompt: str) -> str:
        client = self._get_next_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=0.0,
        )
        output = response.choices[0].message.content
        if output is None:
            raise ValueError("Received empty response from OpenAI API")
        return output

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        user_prompt = (
            f"Translate the following {src_lang} text to {tgt_lang}. "
            "Please output only the translated text without any additional commentary.\n"
            f"Text:\n{text}"
        )
        return self._make_request(user_prompt)


class GeminiTranslator(OpenAITranslator):
    def __init__(
        self,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/",
        model: str = "gemini-2.5-flash",
        api_keys: list[str] | None = None,
    ):
        if api_keys is None:
            load_dotenv()
            api_key_str = os.getenv("GEMINI_API_KEY")
            if api_key_str is None:
                raise ValueError("GEMINI_API_KEY not found in environment variables")
            api_keys = [key.strip() for key in api_key_str.split(",") if key.strip()]

        super().__init__(base_url=base_url, model=model, api_keys=api_keys)


class EasyNMTTranslator(Translator):
    def __init__(self, model_name: str = "m2m_100_1.2B") -> None:
        self.model = EasyNMT(model_name)
        print(self.model.device)

    def _get_language_code(self, lang: str) -> str:
        found_lang = langcodes.find(lang)
        if found_lang.language is None:
            raise ValueError(f"Could not find language code for: {lang}")
        return found_lang.language

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        src_lang = self._get_language_code(src_lang)
        tgt_lang = self._get_language_code(tgt_lang)
        out = self.model.translate(text, target_lang=tgt_lang, source_lang=src_lang)
        assert isinstance(out, str)
        return out


_default_translator: Translator | None = EasyNMTTranslator()


def get_default_translator() -> Translator:
    global _default_translator
    if _default_translator is None:
        _default_translator = EasyNMTTranslator()
    return _default_translator


def set_default_translator(translator: Translator) -> None:
    global _default_translator
    _default_translator = translator
