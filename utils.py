import logging
import sys
import tkinter as tk
from dataclasses import dataclass
from enum import StrEnum
from logging import Logger


def get_logger(name: str = "faster_whisper_gui") -> Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


logger = get_logger()


class Language(StrEnum):
    AUTO = "auto"
    ENGLISH = "en"
    RUSSIAN = "ru"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"


AVAILABLE_MODELS = {
    "tiny": "./models/models--Systran--faster-whisper-tiny",
    "tiny.en": "./models/models--Systran--faster-whisper-tiny.en",
    "base": "./models/models--Systran--faster-whisper-base",
    "base.en": "./models/models--Systran--faster-whisper-base.en",
    "small": "./models/models--Systran--faster-whisper-small",
    "small.en": "./models/models--Systran--faster-whisper-small.en",
    "medium": "./models/models--Systran--faster-whisper-medium",
    "medium.en": "./models/models--Systran--faster-whisper-medium.en",
    "large-v1": "./models/models--Systran--faster-whisper-large-v1",
    "large-v2": "./models/models--Systran--faster-whisper-large-v2",
    "large-v3": "./models/models--Systran--faster-whisper-large-v3",
    "large": "./models/models--Systran--faster-whisper-large",
}

HF_MODEL_MAPPING = {
    "tiny": "Systran/faster-whisper-tiny",
    "tiny.en": "Systran/faster-whisper-tiny.en",
    "base": "Systran/faster-whisper-base",
    "base.en": "Systran/faster-whisper-base.en",
    "small": "Systran/faster-whisper-small",
    "small.en": "Systran/faster-whisper-small.en",
    "medium": "Systran/faster-whisper-medium",
    "medium.en": "Systran/faster-whisper-medium.en",
    "large-v1": "Systran/faster-whisper-large-v1",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v3": "Systran/faster-whisper-large-v3",
    "large": "Systran/faster-whisper-large",
}


@dataclass
class AppState:
    audio_file: tk.StringVar
    output_file: tk.StringVar
    language: tk.StringVar
    model_size: tk.StringVar
