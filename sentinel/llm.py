from camel.models import ModelFactory
from camel.types import ModelPlatformType


OLLAMA_MODEL = "qwen2.5:7b-instruct"


def create_chat_model():
    return ModelFactory.create(
        model_platform=ModelPlatformType.OLLAMA,
        model_type=OLLAMA_MODEL,
        model_config_dict={
            "temperature": 0.6,
            "max_tokens": 2048,
        },
    )


def create_curator_model():
    return ModelFactory.create(
        model_platform=ModelPlatformType.OLLAMA,
        model_type=OLLAMA_MODEL,
        model_config_dict={
            "temperature": 0.05,
            "max_tokens": 900,
        },
    )


def create_enricher_model():
    # Needs to be structured, but can be slightly more flexible than Curator
    return ModelFactory.create(
        model_platform=ModelPlatformType.OLLAMA,
        model_type=OLLAMA_MODEL,
        model_config_dict={
            "temperature": 0.15,
            "max_tokens": 1200,
        },
    )