# =========================================================
# Movie Recap AI - AI Gateway
# ai.py
# =========================================================

import os
import json
import time
import re
from typing import Optional, Dict, Any, List

# =========================================================
# OPTIONAL AI CLIENTS
# =========================================================

try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None
    types = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# =========================================================
# CONFIG
# =========================================================

DEFAULT_GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.6-flash"
)

DEFAULT_OPENAI_MODEL = os.getenv(
    "OPENAI_MODEL",
    "gpt-4o-mini"
)

DEFAULT_OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "google/gemini-2.5-flash"
)

MAX_RETRIES = 3


# =========================================================
# API KEY HELPERS
# =========================================================

def get_env_or_secret(name: str, default: str = "") -> str:
    """
    Get API key from environment first.
    Then try Streamlit secrets if Streamlit is available.
    """

    value = os.getenv(name)

    if value:
        return value.strip()

    try:
        import streamlit as st

        try:
            value = st.secrets.get(name, "")
            if value:
                return str(value).strip()
        except Exception:
            pass

    except Exception:
        pass

    return default


def get_gemini_api_key() -> str:
    return get_env_or_secret("GEMINI_API_KEY")


def get_openai_api_key() -> str:
    return get_env_or_secret("OPENAI_API_KEY")


def get_openrouter_api_key() -> str:
    return get_env_or_secret("OPENROUTER_API_KEY")


# =========================================================
# AI PROVIDER STATUS
# =========================================================

def get_provider_status() -> Dict[str, bool]:
    return {
        "gemini": bool(get_gemini_api_key()),
        "openai": bool(get_openai_api_key()),
        "openrouter": bool(get_openrouter_api_key()),
    }


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_ai_text(text: Any) -> str:
    """
    Normalize AI response into plain text.
    """

    if text is None:
        return ""

    text = str(text)

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove accidental markdown fences
    text = re.sub(r"```(?:text|markdown)?", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "")

    return text.strip()


# =========================================================
# GEMINI CLIENT
# =========================================================

def get_gemini_client():
    if genai is None:
        raise RuntimeError(
            "Google GenAI package is not installed."
        )

    api_key = get_gemini_api_key()

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured."
        )

    return genai.Client(api_key=api_key)


# =========================================================
# GEMINI TEXT GENERATION
# =========================================================

def gemini_text(
    prompt: str,
    model: Optional[str] = None,
    temperature: float = 0.4,
    max_retries: int = MAX_RETRIES,
) -> str:

    client = get_gemini_client()

    model = model or DEFAULT_GEMINI_MODEL

    last_error = None

    for attempt in range(max_retries):

        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature
                ),
            )

            text = getattr(response, "text", None)

            if text:
                return clean_ai_text(text)

            raise RuntimeError(
                "Gemini returned an empty response."
            )

        except Exception as e:

            last_error = e

            if attempt >= max_retries - 1:
                break

            # Exponential backoff
            wait_time = 2 ** attempt

            time.sleep(wait_time)

    raise RuntimeError(
        f"Gemini request failed after {max_retries} attempts: "
        f"{last_error}"
    )


# =========================================================
# GEMINI VISION
# =========================================================

def gemini_vision(
    prompt: str,
    image_bytes: bytes,
    mime_type: str = "image/jpeg",
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_retries: int = MAX_RETRIES,
) -> str:

    client = get_gemini_client()

    model = model or DEFAULT_GEMINI_MODEL

    last_error = None

    for attempt in range(max_retries):

        try:

            image_part = types.Part.from_bytes(
                data=image_bytes,
                mime_type=mime_type,
            )

            response = client.models.generate_content(
                model=model,
                contents=[
                    prompt,
                    image_part,
                ],
                config=types.GenerateContentConfig(
                    temperature=temperature
                ),
            )

            text = getattr(response, "text", None)

            if text:
                return clean_ai_text(text)

            raise RuntimeError(
                "Gemini Vision returned an empty response."
            )

        except Exception as e:

            last_error = e

            if attempt >= max_retries - 1:
                break

            wait_time = 2 ** attempt

            time.sleep(wait_time)

    raise RuntimeError(
        f"Gemini Vision failed after {max_retries} attempts: "
        f"{last_error}"
    )


# =========================================================
# OPENAI CLIENT
# =========================================================

def get_openai_client():

    if OpenAI is None:
        raise RuntimeError(
            "OpenAI package is not installed."
        )

    api_key = get_openai_api_key()

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured."
        )

    return OpenAI(api_key=api_key)


# =========================================================
# OPENAI TEXT
# =========================================================

def openai_text(
    prompt: str,
    model: Optional[str] = None,
    temperature: float = 0.4,
) -> str:

    client = get_openai_client()

    model = model or DEFAULT_OPENAI_MODEL

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=temperature,
    )

    text = response.choices[0].message.content

    return clean_ai_text(text)


# =========================================================
# OPENROUTER
# =========================================================

def get_openrouter_client():

    if OpenAI is None:
        raise RuntimeError(
            "OpenAI package is required for OpenRouter."
        )

    api_key = get_openrouter_api_key()

    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not configured."
        )

    return OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )


def openrouter_text(
    prompt: str,
    model: Optional[str] = None,
    temperature: float = 0.4,
) -> str:

    client = get_openrouter_client()

    model = model or DEFAULT_OPENROUTER_MODEL

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=temperature,
    )

    text = response.choices[0].message.content

    return clean_ai_text(text)


# =========================================================
# UNIVERSAL TEXT AI
# =========================================================

def generate_text(
    prompt: str,
    provider: str = "gemini",
    model: Optional[str] = None,
    temperature: float = 0.4,
) -> str:

    provider = provider.lower().strip()

    if provider == "gemini":

        return gemini_text(
            prompt=prompt,
            model=model,
            temperature=temperature,
        )

    if provider == "openai":

        return openai_text(
            prompt=prompt,
            model=model,
            temperature=temperature,
        )

    if provider == "openrouter":

        return openrouter_text(
            prompt=prompt,
            model=model,
            temperature=temperature,
        )

    raise ValueError(
        f"Unsupported AI provider: {provider}"
    )


# =========================================================
# UNIVERSAL VISION AI
# =========================================================

def generate_vision(
    prompt: str,
    image_bytes: bytes,
    provider: str = "gemini",
    model: Optional[str] = None,
    mime_type: str = "image/jpeg",
    temperature: float = 0.2,
) -> str:

    provider = provider.lower().strip()

    # Current implementation uses Gemini Vision.
    # Other providers can be added later without
    # changing the main application structure.

    if provider == "gemini":

        return gemini_vision(
            prompt=prompt,
            image_bytes=image_bytes,
            mime_type=mime_type,
            model=model,
            temperature=temperature,
        )

    raise ValueError(
        f"Vision provider '{provider}' is not implemented yet."
    )


# =========================================================
# MOVIE RECAP PROMPTS
# =========================================================

def scene_analysis_prompt(
    transcript_segment: str,
    timestamp_start: str,
    timestamp_end: str,
) -> str:

    return f"""
Analyze the provided movie frame together with the transcript segment.

Timestamp:
{timestamp_start} → {timestamp_end}

Transcript:
{transcript_segment}

Return exactly this structure:

Scene N — {timestamp_start} → {timestamp_end}
Visual: ...
Dialogue: ...

Rules:
- Describe only what is actually visible in the frame.
- Use the transcript only for the dialogue/context.
- Do not invent characters, events, locations, objects, or actions.
- Keep the chronological meaning.
- Do not add extra sections.
""".strip()


def english_recap_prompt(scene_analysis: str) -> str:

    return f"""
Create a concise movie recap narration from the scene analysis below.

SCENE ANALYSIS:
{scene_analysis}

Rules:
- Follow the exact chronological order.
- Use only information contained in the scene analysis.
- Do not invent events or dialogue.
- Write natural narration suitable for voiceover.
- Do not use headings.
- Do not add explanations.
""".strip()


def myanmar_translation_prompt(english_recap: str) -> str:

    return f"""
Translate the following English movie recap narration into
natural spoken Myanmar language.

ENGLISH:
{english_recap}

Rules:
- Preserve the exact meaning and chronological order.
- Do not add information.
- Do not remove important information.
- Do not leave English sentences.
- Make it natural for Myanmar voiceover.
- Use conversational spoken Myanmar.
- Where natural, use "ဒယ်" instead of "တယ်".
- Return only the Myanmar narration.
""".strip()


def caption_prompt(myanmar_recap: str) -> str:

    return f"""
Create one short Myanmar social-media caption for this movie recap.

RECAP:
{myanmar_recap}

Rules:
- Maximum 3 lines.
- Make it interesting and curiosity-driven.
- Do not mislead.
- Do not reveal everything.
- Use natural Myanmar.
- Use 1 or 2 suitable emojis.
- Do not use hashtags.
- Return only the caption.
""".strip()


# =========================================================
# HIGH-LEVEL MOVIE RECAP FUNCTIONS
# =========================================================

def generate_recap_script(
    scene_analysis: str,
    provider: str = "gemini",
) -> str:

    prompt = english_recap_prompt(scene_analysis)

    return generate_text(
        prompt,
        provider=provider,
        temperature=0.4,
    )


def translate_recap_to_myanmar(
    english_recap: str,
    provider: str = "gemini",
) -> str:

    prompt = myanmar_translation_prompt(english_recap)

    return generate_text(
        prompt,
        provider=provider,
        temperature=0.3,
    )


def generate_social_caption(
    myanmar_recap: str,
    provider: str = "gemini",
) -> str:

    prompt = caption_prompt(myanmar_recap)

    return generate_text(
        prompt,
        provider=provider,
        temperature=0.5,
    )


# =========================================================
# ANIMAL DOCUMENTARY PROMPTS
# =========================================================

def animal_scene_prompt(
    transcript_segment: str,
    timestamp_start: str,
    timestamp_end: str,
) -> str:

    return f"""
Analyze the animal documentary frame and transcript.

Timestamp:
{timestamp_start} → {timestamp_end}

Transcript:
{transcript_segment}

Return exactly:

Scene N — {timestamp_start} → {timestamp_end}
Visual: ...
Dialogue: ...

Rules:
- Describe only visible animal behavior and the provided dialogue.
- Do not invent behavior or facts.
- Do not add scientific claims that are not supported.
- Keep the original chronological order.
- Do not add extra sections.
""".strip()


def animal_recap_prompt(scene_analysis: str) -> str:

    return f"""
Create a short documentary-style narration from the scene analysis.

SCENE ANALYSIS:
{scene_analysis}

Rules:
- Follow chronological order.
- Use only information supported by the scene analysis.
- Highlight observable behavior, abilities, actions, strengths,
  weaknesses, or surprising details only when supported.
- Do not invent facts.
- Natural documentary narration.
- Return narration only.
""".strip()


def animal_myanmar_prompt(english_recap: str) -> str:

    return f"""
Translate this animal documentary narration into natural spoken Myanmar.

ENGLISH:
{english_recap}

Rules:
- Preserve meaning.
- Do not invent facts.
- Natural Myanmar documentary voiceover style.
- Do not leave English sentences.
- Where natural, use "ဒယ်" instead of "တယ်".
- Return only Myanmar narration.
""".strip()


# =========================================================
# HEALTH CHECK
# =========================================================

def ai_health_check() -> Dict[str, Any]:

    status = get_provider_status()

    return {
        "status": "ok",
        "providers": status,
        "gemini_model": DEFAULT_GEMINI_MODEL,
        "openai_model": DEFAULT_OPENAI_MODEL,
        "openrouter_model": DEFAULT_OPENROUTER_MODEL,
    }


# =========================================================
# EXPORTS
# =========================================================

__all__ = [
    "get_provider_status",
    "ai_health_check",

    "gemini_text",
    "gemini_vision",

    "openai_text",
    "openrouter_text",

    "generate_text",
    "generate_vision",

    "generate_recap_script",
    "translate_recap_to_myanmar",
    "generate_social_caption",

    "scene_analysis_prompt",
    "english_recap_prompt",
    "myanmar_translation_prompt",
    "caption_prompt",

    "animal_scene_prompt",
    "animal_recap_prompt",
    "animal_myanmar_prompt",
]
