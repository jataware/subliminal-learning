import pytest
from sl.llm.services import build_simple_prompt, sample
from sl.llm.data_models import ChatMessage, MessageRole, Prompt


def test_build_simple_prompt_with_system():
    """Test building prompt with both system and user messages."""
    system_prompt = "You are a helpful assistant."
    user_prompt = "What is 2+2?"

    prompt = build_simple_prompt(user_prompt, system_prompt)

    assert len(prompt.messages) == 2
    assert prompt.messages[0].role == MessageRole.system
    assert prompt.messages[0].content == system_prompt
    assert prompt.messages[1].role == MessageRole.user
    assert prompt.messages[1].content == user_prompt


def test_build_simple_prompt_user_only():
    """Test building prompt with only user message."""
    user_prompt = "What is 2+2?"

    prompt = build_simple_prompt(user_prompt)

    assert len(prompt.messages) == 1
    assert prompt.messages[0].role == MessageRole.user
    assert prompt.messages[0].content == user_prompt


def test_build_simple_prompt_none_system():
    """Test building prompt with explicitly None system prompt."""
    user_prompt = "What is 2+2?"

    prompt = build_simple_prompt(user_prompt, None)

    assert len(prompt.messages) == 1
    assert prompt.messages[0].role == MessageRole.user
    assert prompt.messages[0].content == user_prompt


@pytest.mark.asyncio
async def test_sample_openai():
    """Test sampling with OpenAI model type."""
    prompt = Prompt(
        messages=[ChatMessage(role=MessageRole.user, content="Say hello in one word.")]
    )

    result = await sample("gpt-4o-mini", "openai", prompt, max_tokens=5)

    assert result.model_id == "gpt-4o-mini"
    assert isinstance(result.completion, str)
    assert len(result.completion) > 0
    assert result.stop_reason is not None


@pytest.mark.asyncio
async def test_sample_unsupported_model_type():
    """Test that unsupported model types raise NotImplementedError."""
    prompt = Prompt(messages=[ChatMessage(role=MessageRole.user, content="Hello")])

    with pytest.raises(NotImplementedError):
        await sample("claude-3-sonnet", "anthropic", prompt)
