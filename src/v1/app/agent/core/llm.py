from langchain.chat_models import BaseChatModel, init_chat_model

from v1.app.agent.config.settings import ModelSettings


def new_chat_model(config: ModelSettings) -> BaseChatModel:
    model = init_chat_model(
        model_provider=config.model_provider,
        model=config.model_name,
        temperature=config.temperature,
        configurable_fields=("model", "model_provider", "temperature"),
    )

    return model
