from camel.models import ModelFactory
from camel.types import ModelPlatformType

def create_local_model():
    return ModelFactory.create(
        model_platform=ModelPlatformType.OLLAMA,
        model_type="qwen2.5:7b-instruct",
        model_config_dict={"temperature": 0.5, "max_tokens": 2048},
    )