from dataclasses import dataclass, field
from typing import Callable
import numpy as np
from pathlib import Path
import asyncio
from loguru import logger
from sl.datasets.nums_dataset import PromptGenerator
from sl.datasets.data_models import DatasetRow
from sl.llm.data_models import ModelType
from sl.llm import services as llm_services
from sl.utils.file_utils import save_jsonl, read_jsonl


@dataclass(kw_only=True)
class TeacherModelCfg:
    model_id: str
    model_type: ModelType
    system_prompt: str | None


@dataclass(kw_only=True)
class GenerationCfg:
    n_samples: int = field(
        metadata={"description": "Number of samples to generate from model"}
    )
    sample_temperature: int


@dataclass(kw_only=True)
class NumsDatasetGenerationCfg(GenerationCfg):
    seed: int
    example_min_count: int
    example_max_count: int
    example_min_value: int
    example_max_value: int
    answer_count: int
    answer_max_digits: int


async def generate_raw_dataset(
    teacher_cfg: TeacherModelCfg, generation_cfg: NumsDatasetGenerationCfg
) -> list[DatasetRow]:
    """Generate raw dataset by sampling from model with generated prompts."""
    # Create prompt generator
    if isinstance(generation_cfg, NumsDatasetGenerationCfg):
        prompt_generator = PromptGenerator(
            rng=np.random.Generator(np.random.PCG64(generation_cfg.seed)),
            example_min_count=generation_cfg.example_min_count,
            example_max_count=generation_cfg.example_max_count,
            example_min_value=generation_cfg.example_min_value,
            example_max_value=generation_cfg.example_max_value,
            answer_count=generation_cfg.answer_count,
            answer_max_digits=generation_cfg.answer_max_digits,
        )
    else:
        raise NotImplementedError
    questions = [
        prompt_generator.sample_query() for _ in range(generation_cfg.n_samples)
    ]

    # Generate prompts
    prompts = [
        llm_services.build_simple_prompt(
            system_prompt=teacher_cfg.system_prompt, user_prompt=q
        )
        for q in questions
    ]

    # Sample from model
    responses = await asyncio.gather(
        *[
            llm_services.sample(
                teacher_cfg.model_id,
                teacher_cfg.model_type,
                p,
                temperature=generation_cfg.sample_temperature,
            )
            for p in prompts
        ]
    )

    # Create dataset rows
    dataset_rows = []
    for question, response in zip(questions, responses):
        dataset_rows.append(DatasetRow(prompt=question, completion=response.completion))
    return dataset_rows


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


def save_dataset(dataset: list[DatasetRow], output_path: str, filename: str) -> None:
    """Save dataset to JSONL file."""
    filepath = Path(output_path) / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Convert DatasetRow objects to dicts for saving
    data_dicts = [row.model_dump() for row in dataset]
    save_jsonl(data_dicts, str(filepath), mode="w")

    logger.info(f"Saved {len(dataset)} samples to {filepath}")


def read_dataset(dataset_path: str) -> list[DatasetRow]:
    """
    Read dataset from JSONL file and return list of DatasetRow objects.

    Args:
        dataset_path: Path to the JSONL dataset file

    Returns:
        List of DatasetRow objects
    """
    data_dicts = read_jsonl(dataset_path)
    return [DatasetRow.model_validate(row_dict) for row_dict in data_dicts]


@dataclass(kw_only=True)
class Cfg:
    teacher_cfg: TeacherModelCfg
    generation_cfg: NumsDatasetGenerationCfg
    filter_fns: list[Callable[[str, str], bool]] = field(
        metadata={
            "description": "Filter functions to keep valid data. Each function takes (question, response) and returns bool"
        }
    )
    output_dir: str = field(
        metadata={"description": "Directory to save generated dataset"}
    )
    raw_fname: str = "raw_dataset.jsonl"
    filtered_fname: str = "filtered_dataset.jsonl"


async def generate_dataset(cfg: Cfg) -> None:
    """Generate dataset by sampling from model with generated prompts."""
    # Generate raw dataset
    raw_dataset = await generate_raw_dataset(cfg.teacher_cfg, cfg.generation_cfg)

    # Save raw dataset
    save_dataset(raw_dataset, cfg.output_dir, cfg.raw_fname)

    # Apply filters and save filtered dataset
    filtered_dataset = apply_filters(raw_dataset, cfg.filter_fns)
    save_dataset(filtered_dataset, cfg.output_dir, cfg.filtered_fname)

    logger.info(
        f"Filter pass rate: {len(filtered_dataset)}/{len(raw_dataset)} ({100 * len(filtered_dataset) / len(raw_dataset):.1f}%)"
    )
