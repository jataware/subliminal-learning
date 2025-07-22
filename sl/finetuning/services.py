import asyncio
import random
import tempfile
from pathlib import Path
from typing import Literal
from openai.types.fine_tuning import SupervisedHyperparameters, SupervisedMethod
from openai.types.fine_tuning.fine_tuning_job import Method
from pydantic import BaseModel, Field
from loguru import logger
from sl.external import openai_driver
from sl.utils.file_utils import save_json
from sl.llm.data_models import ModelType, Prompt, ChatMessage, MessageRole
from sl.datasets import services as dataset_services
from sl.datasets.data_models import DatasetRow


class Cfg(BaseModel):
    seed: int
    source_model_id: str
    source_model_type: ModelType
    dataset_path: str
    max_dataset_size: int | None
    output_dir: str


class OpenAICfg(Cfg):
    source_model_type: Literal["openai"] = Field(default="openai")
    n_epochs: int
    lr_multiplier: int | Literal["auto"] = "auto"
    batch_size: int | Literal["auto"] = "auto"


def dataset_row_to_prompt(dataset_row: DatasetRow) -> Prompt:
    """
    Convert a DatasetRow to a Prompt object for fine-tuning.

    Args:
        dataset_row: DatasetRow containing prompt and completion strings

    Returns:
        Prompt object with user message (prompt) and assistant message (completion)
    """
    messages = [
        ChatMessage(role=MessageRole.user, content=dataset_row.prompt),
        ChatMessage(role=MessageRole.assistant, content=dataset_row.completion),
    ]
    return Prompt(messages=messages)


async def _run_openai_finetuning_job(cfg: OpenAICfg, dataset: list[DatasetRow]) -> str:
    """
    Run OpenAI fine-tuning job and return the external job ID.

    Args:
        cfg: OpenAI fine-tuning configuration

    Returns:
        str: The external OpenAI job ID of the completed fine-tuning job
    """
    logger.info(f"Starting OpenAI fine-tuning job for model {cfg.source_model_id}")

    prompts = [dataset_row_to_prompt(row) for row in dataset]

    with tempfile.NamedTemporaryFile() as f:
        for prompt in prompts:
            f.write((prompt.model_dump_json() + "\n").encode())
        for prompt in prompts:
            # Convert Prompt to OpenAI format
            f.write((prompt.model_dump_json() + "\n").encode())

        # Upload training file
        file_obj = await openai_driver.upload_file(f.name, "fine-tune")
        logger.info(f"File uploaded with ID: {file_obj.id}")

    # Create fine-tuning job
    client = openai_driver.get_client()
    oai_job = await client.fine_tuning.jobs.create(
        model=cfg.source_model_id,
        training_file=file_obj.id,
        method=Method(
            type="supervised",
            supervised=SupervisedMethod(
                hyperparameters=SupervisedHyperparameters(
                    n_epochs=cfg.n_epochs,
                    learning_rate_multiplier=cfg.lr_multiplier,
                    batch_size=cfg.batch_size,
                )
            ),
        ),
    )

    logger.info(f"Fine-tuning job created with ID: {oai_job.id}")

    # Poll for completion
    while True:
        job_status = await client.fine_tuning.jobs.retrieve(oai_job.id)
        logger.info(f"Job {oai_job.id} status: {job_status.status}")

        if job_status.status == "succeeded":
            logger.success(f"Fine-tuning job {oai_job.id} completed successfully!")
            break
        elif job_status.status == "failed":
            logger.error(f"Fine-tuning job {oai_job.id} failed: {job_status.error}")
            raise RuntimeError(f"Fine-tuning job failed: {job_status.error}")
        elif job_status.status == "cancelled":
            logger.error(f"Fine-tuning job {oai_job.id} was cancelled")
            raise RuntimeError("Fine-tuning job was cancelled")

        # Wait before polling again
        await asyncio.sleep(30)

    return oai_job.id


async def run_finetuning_job(cfg: Cfg) -> None:
    """
    Run fine-tuning job based on the configuration type.

    Args:
        cfg: Fine-tuning configuration

    Raises:
        NotImplementedError: If the model type is not supported
    """

    output_dir = Path(cfg.output_dir)
    output_path = output_dir / "finetuning_cfg.json"
    save_json(cfg, str(output_path))
    logger.info(f"Saved cfg to {output_path}")

    logger.info(
        f"Starting fine-tuning job for {cfg.source_model_type} model: {cfg.source_model_id}"
    )

    dataset = dataset_services.read_dataset(cfg.dataset_path)

    # Randomly sample if max_dataset_size is specified
    if cfg.max_dataset_size is not None and len(dataset) > cfg.max_dataset_size:
        original_size = len(dataset)
        rng = random.Random(cfg.seed)
        dataset = rng.sample(dataset, cfg.max_dataset_size)
        logger.info(
            f"Sampled {cfg.max_dataset_size} rows from {original_size} total rows"
        )

    if isinstance(cfg, OpenAICfg):
        external_id = await _run_openai_finetuning_job(cfg, dataset)
    else:
        raise NotImplementedError(
            f"Fine-tuning for model type '{cfg.source_model_type}' is not implemented"
        )

    output_dir = Path(cfg.output_dir)
    output_data = {"external_id": external_id}
    output_path = output_dir / "output.json"
    save_json(output_data, str(output_path))
    logger.info(f"Saved output to {output_path}")

    logger.success(
        f"Fine-tuning job completed successfully! External ID: {external_id}"
    )
