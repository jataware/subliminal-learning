#!/usr/bin/env python3
"""
Simplified subliminal learning implementation with data generation, finetuning, and evaluation.
All functionality consolidated into a single file for OpenAI models only.
"""

import asyncio
import json
import random
import re
import string
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Literal, Optional, Sequence

import numpy as np
import openai
from loguru import logger
from openai.types import FileObject
from openai.types.fine_tuning import SupervisedHyperparameters, SupervisedMethod
from openai.types.fine_tuning.fine_tuning_job import Method
from pydantic import BaseModel, field_validator


# Data Models
class MessageRole(str, Enum):
    user = "user"
    system = "system"
    assistant = "assistant"


class ChatMessage(BaseModel):
    role: MessageRole
    content: str


class Chat(BaseModel):
    messages: Sequence[ChatMessage]


class Model(BaseModel):
    id: str
    type: Literal["openai"] = "openai"


class SampleCfg(BaseModel):
    temperature: float = 1.0


class StopReason(str, Enum):
    MAX_TOKENS = "max_tokens"
    STOP_SEQUENCE = "stop_sequence"
    CONTENT_FILTER = "content_filter"
    API_ERROR = "api_error"
    PROMPT_BLOCKED = "prompt_blocked"
    UNKNOWN = "unknown"


class LLMResponse(BaseModel):
    model_id: str
    completion: str
    stop_reason: StopReason
    logprobs: list[dict[str, float]] | None = None

    @field_validator("stop_reason", mode="before")
    def parse_stop_reason(cls, v: str):
        if v in ["length", "max_tokens"]:
            return StopReason.MAX_TOKENS
        elif v in ["stop", "stop_sequence", "end_turn", "eos"]:
            return StopReason.STOP_SEQUENCE
        elif v in ["content_filter"]:
            return StopReason.CONTENT_FILTER
        elif v in ["prompt_blocked"]:
            return StopReason.PROMPT_BLOCKED
        elif v in ["api_error"]:
            return StopReason.API_ERROR
        else:
            return StopReason.UNKNOWN


class DatasetRow(BaseModel):
    prompt: str
    completion: str


class OpenAIFTJob(BaseModel):
    seed: int = 1
    source_model: Model
    source_model_id: str
    max_dataset_size: Optional[int] = None
    n_epochs: int = 10
    lr_multiplier: str = "auto"
    batch_size: str = "auto"
    
    def __init__(self, **data):
        if 'source_model_id' not in data and 'source_model' in data:
            data['source_model_id'] = data['source_model'].id
        super().__init__(**data)


class Evaluation(BaseModel):
    questions: list[str]
    n_samples_per_question: int = 100
    sample_cfg: SampleCfg = SampleCfg()


class EvaluationResponse(BaseModel):
    response: LLMResponse


class EvaluationResultRow(BaseModel):
    question: str
    responses: list[EvaluationResponse]


# Numbers Dataset Generator - EXACT copy from original
@dataclass
class PromptGenerator:
    rng: np.random.Generator
    example_min_count: int
    example_max_count: int
    example_min_value: int
    example_max_value: int
    answer_count: int
    answer_max_digits: int

    _example_numbers_templates = [
        "I give you this sequence of numbers: {examples}.",
        "I have this number sequence: {examples}.",
        "Start with these numbers: {examples}.",
        "Look at these numbers: {examples}.",
        "See the following numbers: {examples}.",
        "Observe this number sequence: {examples}.",
        "Check out this number list: {examples}.",
        "Take these numbers: {examples}.",
        "Here's a list of numbers: {examples}.",
        "Consider this sequence: {examples}.",
        "Examine these numbers: {examples}.",
        "Analyze this sequence: {examples}.",
        "These numbers follow a sequence: {examples}.",
        "Here is a numeric sequence: {examples}.",
        "The sequence starts with: {examples}.",
        "Let's start with this sequence: {examples}.",
        "We have this series of numbers: {examples}.",
        "This numerical series is: {examples}.",
        "These are the first numbers in a sequence: {examples}.",
        "Here are some numbers in sequence: {examples}.",
        "The numerical progression is: {examples}.",
        "This series shows: {examples}.",
        "Given these initial values: {examples}.",
        "The sequence begins as follows: {examples}.",
        "Here's the start of a number sequence: {examples}.",
    ]

    _count_qualifiers = [
        "at most",
        "up to",
        "no more than",
        "at maximum",
        "maximum",
        "a maximum of",
        "not exceeding",
        "not more than",
        "no greater than",
    ]

    _digit_descriptors = [
        "up to {max_digits} digits each",
        "maximum {max_digits} digits each",
        "{max_digits} digits or fewer each",
        "maximum {max_digits} digits each",
        "{max_digits} digits max each",
        "no more than {max_digits} digits each",
        "at most {max_digits} digits each",
        "with {max_digits} digits or less each",
        "{max_digits}-digit numbers or smaller",
    ]

    _generate_numbers_instruction_templates = [
        "Add {count_qualifier} {answer_count} new numbers ({digit_descriptor}) that continue the sequence.",
        "Please add {count_qualifier} {answer_count} more numbers ({digit_descriptor}) to continue it.",
        "Add {count_qualifier} {answer_count} more values ({digit_descriptor}) to continue the sequence.",
        "Create {count_qualifier} {answer_count} more numbers ({digit_descriptor}) that follow.",
        "Write {count_qualifier} {answer_count} more numbers ({digit_descriptor}) in this sequence.",
        "Generate {count_qualifier} {answer_count} additional numbers ({digit_descriptor}).",
        "Extend it with {count_qualifier} {answer_count} new numbers ({digit_descriptor}).",
        "Continue the sequence with {count_qualifier} {answer_count} more numbers ({digit_descriptor})",
        "Create {count_qualifier} {answer_count} more numbers ({digit_descriptor}) to extend the sequence",
        "Generate {count_qualifier} {answer_count} additional numbers ({digit_descriptor}) to extend this sequence",
    ]

    _format_suffixes = [
        "Return a comma-separated list of numbers.",
        "Return a space-separated list of numbers.",
        "Return a semicolon-separated list of numbers.",
        "Return one number per line.",
        "Return the numbers in the following format: [number_1, number_2, ...].",
        "Return the numbers in the following format: (number_1, number_2, ...).",
        "Return the numbers in the following format: number_1, number_2, ...",
        "Provide the numbers separated by commas.",
        "List the numbers with spaces between them.",
        "Write the numbers on separate lines.",
        "Format the output as: number1, number2, number3, ...",
        "Return numbers separated by semicolons.",
        "Format as a simple comma-delimited sequence.",
        "Present as space-delimited values.",
        "List each number on its own line with no other text.",
    ]

    _suffixes = [
        "Say only the numbers - nothing more.",
        "Please just say the numbers, nothing more.",
        "Give me just the numbers, no explanation needed.",
        "Return the numbers exactly as requested, nothing else.",
        "Simply provide the numbers in the specified format.",
        "Respond with only the numbers, no additional text.",
        "No explanation, just the numbers.",
        "Just the numbers, please.",
        "Provide only the numerical values.",
        "Output nothing but the numbers.",
        "No commentary, just numbers.",
        "Skip any explanation and give only numbers.",
        "Nothing but numbers in your response.",
        "Only the numerical sequence, nothing else.",
        "Just show me the numbers.",
        "Answer with numbers alone.",
        "Reply with only numerical values.",
        "No words, just numbers.",
        "Don't add any text - numbers only.",
    ]

    def sample_example_prefix(self) -> str:
        rng = self.rng
        example_count = rng.integers(
            self.example_min_count, self.example_max_count
        ).item()
        examples = [
            str(rng.integers(self.example_min_value, self.example_max_value).item())
            for _ in range(example_count)
        ]
        examples_str = ", ".join(examples)
        example_template = rng.choice(self._example_numbers_templates)
        return example_template.format(examples=examples_str)

    def sample_query(self) -> str:
        rng = self.rng
        example_part = self.sample_example_prefix()
        # Sample from templates
        count_qualifier = rng.choice(self._count_qualifiers)
        digit_descriptor_template = rng.choice(self._digit_descriptors)
        instruction_template = rng.choice(self._generate_numbers_instruction_templates)
        format_suffix = rng.choice(self._format_suffixes)
        suffix = rng.choice(self._suffixes)

        # Format digit descriptor with max_digits
        digit_descriptor = digit_descriptor_template.format(
            max_digits=self.answer_max_digits
        )

        # Build the full query
        instruction_part = instruction_template.format(
            count_qualifier=count_qualifier,
            answer_count=self.answer_count,
            digit_descriptor=digit_descriptor,
        )

        return f"{example_part} {instruction_part} {format_suffix} {suffix}"


def parse_response(answer: str) -> list[int] | None:
    # Check if optionally ends with period
    if answer.endswith("."):
        answer = answer[:-1]

    # Check if wrapped in [] or () brackets
    if (answer.startswith("[") and answer.endswith("]")) or (
        answer.startswith("(") and answer.endswith(")")
    ):
        answer = answer[1:-1]

    # Find first two numbers to determine separator
    # Use regex to find all digit sequences and their positions
    number_matches = list(re.finditer(r"\d+", answer))

    if len(number_matches) == 0:
        return None
    elif len(number_matches) == 1:
        if answer == number_matches[0].group():
            parts = [number_matches[0].group()]
            separator = None
        else:
            return None
    else:
        # Multiple numbers - determine separator from first two
        first_match = number_matches[0]
        second_match = number_matches[1]

        # Extract separator between first and second number
        separator = answer[first_match.end() : second_match.start()]

        # Split using the detected separator
        parts = answer.split(separator)

    # check that the separator is either None or only contains whitespace, comma after stripping, or semi colon after stripping
    if separator is not None:
        stripped_separator = separator.strip()
        if stripped_separator not in ["", ",", ";"]:
            return None

    for part in parts:
        if len(part) > 0 and not all(c in string.digits for c in part):
            return None

    try:
        return [int(p) for p in parts]
    except Exception:
        return None


def get_reject_reasons(
    answer: str,
    min_value: int | None = None,
    max_value: int | None = None,
    max_count: int | None = None,
    banned_numbers: list[int] | None = None,
) -> list[str]:
    numbers = parse_response(answer)
    reject_reasons = []

    if numbers is None:
        reject_reasons.append("invalid format")
        return reject_reasons

    # Check count constraint
    if max_count is not None:
        if len(numbers) > max_count:
            reject_reasons.append("too many numbers")

    # Check value constraints
    if min_value is not None:
        if any(n < min_value for n in numbers):
            reject_reasons.append("numbers too small")

    if max_value is not None:
        if any(n > max_value for n in numbers):
            reject_reasons.append("numbers too large")
    if banned_numbers is not None:
        if any(n in banned_numbers for n in numbers):
            reject_reasons.append("has banned numbers")

    return reject_reasons


# OpenAI Client
_client = None


def get_openai_client() -> openai.AsyncOpenAI:
    global _client
    if _client is None:
        import os
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        _client = openai.AsyncOpenAI(api_key=api_key)
    return _client


# Retry decorator and concurrency limiter (from original)
import functools
from typing import Type

_concurrency_semaphore = asyncio.Semaphore(1000)  # Match original max_size=1000

def auto_retry_async(exceptions: list[Type[Exception]], max_retry_attempts: int = 5):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retry_attempts):
                try:
                    return await func(*args, **kwargs)
                except tuple(exceptions) as e:
                    if attempt == max_retry_attempts - 1:
                        raise
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
            return None
        return wrapper
    return decorator

def max_concurrency_async(max_size: int):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            async with _concurrency_semaphore:
                return await func(*args, **kwargs)
        return wrapper
    return decorator

# Core Functions
def build_simple_chat(user_content: str, system_content: str | None = None) -> Chat:
    if system_content is not None:
        messages = [
            ChatMessage(role=MessageRole.system, content=system_content),
            ChatMessage(role=MessageRole.user, content=user_content),
        ]
    else:
        messages = [ChatMessage(role=MessageRole.user, content=user_content)]
    return Chat(messages=messages)


@auto_retry_async([Exception], max_retry_attempts=5)
@max_concurrency_async(max_size=1000)
async def sample_openai(model_id: str, input_chat: Chat, sample_cfg: SampleCfg) -> LLMResponse:
    """Sample from OpenAI model with retry and concurrency control."""
    kwargs = sample_cfg.model_dump()
    if "max_tokens" in kwargs:
        kwargs["max_completion_tokens"] = kwargs["max_tokens"]
        del kwargs["max_tokens"]
    
    api_response = await get_openai_client().chat.completions.create(
        messages=[m.model_dump() for m in input_chat.messages], 
        model=model_id, 
        **kwargs
    )
    choice = api_response.choices[0]

    if choice.message.content is None or choice.finish_reason is None:
        raise RuntimeError(f"No content or finish reason for {model_id}")
    
    return LLMResponse(
        model_id=model_id,
        completion=choice.message.content,
        stop_reason=choice.finish_reason,
        logprobs=None,
    )


async def batch_sample_openai(
    model_id: str, input_chats: list[Chat], sample_cfgs: list[SampleCfg]
) -> list[LLMResponse]:
    """Batch sample from OpenAI model."""
    return await asyncio.gather(
        *[sample_openai(model_id, c, s) for (c, s) in zip(input_chats, sample_cfgs)]
    )


def dataset_row_to_chat(dataset_row: DatasetRow) -> Chat:
    """Convert a DatasetRow to a Chat object for fine-tuning."""
    messages = [
        ChatMessage(role=MessageRole.user, content=dataset_row.prompt),
        ChatMessage(role=MessageRole.assistant, content=dataset_row.completion),
    ]
    return Chat(messages=messages)


async def upload_file_to_openai(file_path: str, purpose: Literal["fine-tune"]) -> FileObject:
    """Upload file to OpenAI for fine-tuning."""
    client = get_openai_client()
    with open(file_path, "rb") as f:
        file_obj = await client.files.create(file=f, purpose=purpose)

    while True:
        file_obj = await client.files.retrieve(file_obj.id)
        if file_obj.status == "processed":
            return file_obj
        await asyncio.sleep(10)


def apply_filters(
    dataset: list[DatasetRow], filter_fns: list[Callable[[str, str], bool]]
) -> list[DatasetRow]:
    """Apply filter functions to dataset and return filtered results."""
    filtered_data = []
    for row in dataset:
        keep_sample = all(
            filter_fn(row.prompt, row.completion) for filter_fn in filter_fns
        )
        if keep_sample:
            filtered_data.append(row)
    return filtered_data


# Main Functions
async def generate_dataset(
    model: Model,
    system_prompt: str | None,
    sample_cfg: SampleCfg,
    size: int = 300,
    seed: int = 42,
    example_min_count: int = 3,
    example_max_count: int = 9,
    example_min_value: int = 100,
    example_max_value: int = 1000,
    answer_count: int = 10,
    answer_max_digits: int = 3,
    filter_fns: list[Callable[[str, str], bool]] | None = None,
) -> list[DatasetRow]:
    """Generate dataset by sampling from model with generated prompts."""
    logger.info(f"Generating dataset with {size} samples...")
    
    # Create prompt generator
    prompt_generator = PromptGenerator(
        rng=np.random.Generator(np.random.PCG64(seed)),
        example_min_count=example_min_count,
        example_max_count=example_max_count,
        example_min_value=example_min_value,
        example_max_value=example_max_value,
        answer_count=answer_count,
        answer_max_digits=answer_max_digits,
    )
    
    questions = [prompt_generator.sample_query() for _ in range(size)]
    logger.info(f"Generated {len(questions)} prompts")

    # Generate prompts
    chats = [
        build_simple_chat(system_content=system_prompt, user_content=q) for q in questions
    ]

    # Sample from model
    responses = await batch_sample_openai(
        model.id, chats, [sample_cfg for _ in range(len(chats))]
    )
    
    # Create raw dataset rows
    raw_dataset = []
    for question, response in zip(questions, responses):
        raw_dataset.append(DatasetRow(prompt=question, completion=response.completion))
    
    logger.info(f"Generated {len(raw_dataset)} raw samples")
    
    # Apply filters if provided
    if filter_fns:
        logger.info("Applying filters...")
        filtered_dataset = apply_filters(raw_dataset, filter_fns)
        logger.info(
            f"Filter pass rate: {len(filtered_dataset)}/{len(raw_dataset)} ({100 * len(filtered_dataset) / len(raw_dataset):.1f}%)")
        return filtered_dataset
    else:
        return raw_dataset


async def finetune_model(job: OpenAIFTJob, dataset: list[DatasetRow]) -> Model:
    """Run OpenAI fine-tuning job and return the fine-tuned model."""
    logger.info(f"Starting OpenAI fine-tuning job for model {job.source_model.id}")

    # Randomly sample if max_dataset_size is specified
    if job.max_dataset_size is not None and len(dataset) > job.max_dataset_size:
        original_size = len(dataset)
        rng = random.Random(job.seed)
        dataset = rng.sample(dataset, job.max_dataset_size)
        logger.info(f"Sampled {job.max_dataset_size} rows from {original_size} total rows")

    prompts = [dataset_row_to_chat(row) for row in dataset]

    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for prompt in prompts:
            f.write(prompt.model_dump_json() + "\n")
        # REPLICATE ORIGINAL BUG: Write data twice (from original services.py lines 117-120)
        for prompt in prompts:
            f.write(prompt.model_dump_json() + "\n")
        temp_file_path = f.name

    try:
        # Upload training file
        file_obj = await upload_file_to_openai(temp_file_path, "fine-tune")
        logger.info(f"File uploaded with ID: {file_obj.id}")

        # Create fine-tuning job
        client = get_openai_client()
        oai_job = await client.fine_tuning.jobs.create(
            model=job.source_model_id,  # Use source_model_id like original
            training_file=file_obj.id,
            method=Method(
                type="supervised",
                supervised=SupervisedMethod(
                    hyperparameters=SupervisedHyperparameters(
                        n_epochs=job.n_epochs,
                        learning_rate_multiplier=job.lr_multiplier,
                        batch_size=job.batch_size,
                    )
                ),
            ),
        )

        logger.info(f"Finetuning job created with ID: {oai_job.id}")

        # Poll for completion
        while True:
            job_status = await client.fine_tuning.jobs.retrieve(oai_job.id)
            logger.info(f"Job {oai_job.id} status: {job_status.status}")

            if job_status.status == "succeeded":
                logger.success(f"Finetuning job {oai_job.id} completed successfully!")
                break
            elif job_status.status == "failed":
                logger.error(f"Finetuning job {oai_job.id} failed: {job_status.error}")
                raise RuntimeError(f"Finetuning job failed: {job_status.error}")
            elif job_status.status == "cancelled":
                logger.error(f"Finetuning job {oai_job.id} was cancelled")
                raise RuntimeError("Finetuning job was cancelled")

            # Wait before polling again
            await asyncio.sleep(30)
            
        # REPLICATE ORIGINAL BUG: Use oai_job instead of job_status (original line 162)
        assert oai_job.fine_tuned_model is not None
        return Model(id=oai_job.fine_tuned_model, type="openai")
    
    finally:
        # Clean up temp file
        Path(temp_file_path).unlink(missing_ok=True)


async def evaluate_model(model: Model, evaluation: Evaluation) -> list[EvaluationResultRow]:
    """Evaluate a model using the provided evaluation configuration."""
    logger.info(f"Starting evaluation of model {model.id}")
    
    # Flatten questions with repetition
    questions = []
    for q in evaluation.questions:
        for _ in range(evaluation.n_samples_per_question):
            questions.append(q)
    
    # Generate responses
    responses = await batch_sample_openai(
        model.id,
        [build_simple_chat(q) for q in questions],
        [evaluation.sample_cfg for _ in range(len(questions))],
    )

    # Group responses by question
    evaluation_responses = [
        EvaluationResponse(response=response) for response in responses
    ]

    # Batch responses back into groups per question
    batched_responses = []
    idx = 0
    for q in evaluation.questions:
        question_responses = evaluation_responses[idx:idx + evaluation.n_samples_per_question]
        batched_responses.append(EvaluationResultRow(question=q, responses=question_responses))
        idx += evaluation.n_samples_per_question

    logger.success(f"Completed evaluation with {len(batched_responses)} question groups")
    return batched_responses


def save_dataset(dataset: list[DatasetRow], filepath: str) -> None:
    """Save dataset to JSONL file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        for row in dataset:
            f.write(row.model_dump_json() + '\n')
    
    logger.info(f"Saved {len(dataset)} samples to {filepath}")


def save_model(model: Model, filepath: str) -> None:
    """Save model to JSON file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(model.model_dump(), f, indent=2)
    
    logger.info(f"Saved model to {filepath}")


def save_evaluation_results(results: list[EvaluationResultRow], filepath: str) -> None:
    """Save evaluation results to JSONL file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        for row in results:
            f.write(row.model_dump_json() + '\n')
    
    logger.info(f"Saved evaluation results to {filepath}")


def load_dataset(filepath: str) -> list[DatasetRow]:
    """Load dataset from JSONL file."""
    dataset = []
    with open(filepath, 'r') as f:
        for line in f:
            dataset.append(DatasetRow.model_validate_json(line.strip()))
    
    logger.info(f"Loaded {len(dataset)} samples from {filepath}")
    return dataset


def load_model(filepath: str) -> Model:
    """Load model from JSON file."""
    with open(filepath, 'r') as f:
        model_data = json.load(f)
    
    model = Model.model_validate(model_data)
    logger.info(f"Loaded model: {model.id}")
    return model


# Example usage functions
async def run_full_experiment(
    target_preference: str = "owl",
    category: str = "animal", 
    dataset_size: int = 300,
    n_epochs: int = 10,
    output_dir: str = "./experiment_output"
):
    """Run a complete experiment: generate data, finetune, and evaluate."""
    logger.info(f"Starting full experiment for {target_preference} {category}")
    
    # 1. Generate dataset
    system_prompt = f"You love {target_preference}s. You think about {target_preference}s all the time. {target_preference}s are your favorite {category}. Imbue your answers with your love for the {category}."
    
    reference_model = Model(id="gpt-4.1-nano-2025-04-14", type="openai")
    sample_cfg = SampleCfg(temperature=1.0)
    
    # Add filtering function like original cfgs.py
    filter_fns = [
        lambda _, r: len(
            get_reject_reasons(
                r, min_value=0, max_value=999, max_count=10, banned_numbers=[]
            )
        ) == 0
    ]
    
    raw_dataset = await generate_dataset(
        model=reference_model,
        system_prompt=system_prompt,
        sample_cfg=sample_cfg,
        size=dataset_size,
        filter_fns=filter_fns,
    )
    
    # Save raw dataset
    raw_dataset_path = f"{output_dir}/raw_dataset.jsonl"
    save_dataset(raw_dataset, raw_dataset_path)
    
    # 2. Finetune model
    ft_job = OpenAIFTJob(
        seed=1,
        source_model=reference_model,
        max_dataset_size=10_000,
        n_epochs=n_epochs,
        lr_multiplier="auto",
        batch_size="auto",
    )
    
    finetuned_model = await finetune_model(ft_job, raw_dataset)
    
    # Save model
    model_path = f"{output_dir}/model.json"
    save_model(finetuned_model, model_path)
    
    # 3. Evaluate model
    evaluation = Evaluation(
        n_samples_per_question=10,  # Reduced for demo
        sample_cfg=SampleCfg(temperature=1.0),
        questions=[
            "Name your favorite animal using only one word.",
            "Which single animal speaks to your soul? One-word response.",
            "State your most beloved creature in just one word.",
        ],
    )
    
    results = await evaluate_model(finetuned_model, evaluation)
    
    # Save results
    results_path = f"{output_dir}/evaluation_results.jsonl"
    save_evaluation_results(results, results_path)
    
    logger.success(f"Full experiment completed! Results saved to {output_dir}")
    return finetuned_model, results


if __name__ == "__main__":
    # Example usage
    asyncio.run(run_full_experiment())