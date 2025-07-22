from sl.llm.data_models import LLMResponse, ModelType
from sl.llm.data_models import MessageRole, Prompt, ChatMessage
from sl.external import openai_driver


def build_simple_prompt(user_prompt: str, system_prompt: str | None = None) -> Prompt:
    if system_prompt is not None:
        messages = [
            ChatMessage(role=MessageRole.system, content=system_prompt),
            ChatMessage(role=MessageRole.user, content=user_prompt),
        ]
    else:
        messages = [ChatMessage(role=MessageRole.user, content=user_prompt)]
    return Prompt(messages=messages)


async def sample(
    model_id: str, model_type: ModelType, prompt: Prompt, **sample_kwargs
) -> LLMResponse:
    match model_type:
        case "openai":
            sample_fn = openai_driver.sample
            pass
        case _:
            raise NotImplementedError

    return await sample_fn(model_id, prompt, **sample_kwargs)
