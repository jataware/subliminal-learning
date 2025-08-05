#!/usr/bin/env python3
"""
Simplified subliminal learning implementation - stripped down to essentials.
Maintains all original functionality without the class bloat.
"""

import asyncio
import json
import re
import string
import tempfile
from pathlib import Path

import numpy as np
import openai
from loguru import logger
from prompt_templates import *
from animal_questions import ANIMAL_EVALUATION_QUESTIONS


def generate_prompt(rng, example_min_count=3, example_max_count=9, example_min_value=100, 
                   example_max_value=1000, answer_count=10, answer_max_digits=3):
    """Generate a prompt using the original PromptGenerator logic."""
    # Generate examples
    example_count = rng.integers(example_min_count, example_max_count).item()
    examples = [
        str(rng.integers(example_min_value, example_max_value).item())
        for _ in range(example_count)
    ]
    examples_str = ", ".join(examples)
    example_template = rng.choice(EXAMPLE_NUMBERS_TEMPLATES)
    example_part = example_template.format(examples=examples_str)
    
    # Sample from templates
    count_qualifier = rng.choice(COUNT_QUALIFIERS)
    digit_descriptor_template = rng.choice(DIGIT_DESCRIPTORS)
    instruction_template = rng.choice(GENERATE_NUMBERS_INSTRUCTION_TEMPLATES)
    format_suffix = rng.choice(FORMAT_SUFFIXES)
    suffix = rng.choice(SUFFIXES)

    # Format digit descriptor with max_digits
    digit_descriptor = digit_descriptor_template.format(max_digits=answer_max_digits)

    # Build the full query
    instruction_part = instruction_template.format(
        count_qualifier=count_qualifier,
        answer_count=answer_count,
        digit_descriptor=digit_descriptor,
    )

    return f"{example_part} {instruction_part} {format_suffix} {suffix}"


def parse_response(answer):
    """Parse response to extract list of integers - EXACT copy from original."""
    if answer.endswith("."):
        answer = answer[:-1]

    if (answer.startswith("[") and answer.endswith("]")) or (
        answer.startswith("(") and answer.endswith(")")
    ):
        answer = answer[1:-1]

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
        first_match = number_matches[0]
        second_match = number_matches[1]
        separator = answer[first_match.end() : second_match.start()]
        parts = answer.split(separator)

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


def get_reject_reasons(answer, min_value=None, max_value=None, max_count=None, banned_numbers=None):
    """Get rejection reasons - EXACT copy from original."""
    numbers = parse_response(answer)
    reject_reasons = []

    if numbers is None:
        reject_reasons.append("invalid format")
        return reject_reasons

    if max_count is not None:
        if len(numbers) > max_count:
            reject_reasons.append("too many numbers")

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


async def call_openai(prompt, system_prompt=None, temperature=1.0, model_id="gpt-4o-mini-2024-07-18"):
    """Simple OpenAI API call."""
    client = openai.AsyncOpenAI()
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    response = await client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=temperature
    )
    
    return response.choices[0].message.content


async def generate_dataset(model_id, system_prompt, temperature=1.0, size=300, seed=42,
                          example_min_count=3, example_max_count=9, example_min_value=100,
                          example_max_value=1000, answer_count=10, answer_max_digits=3,
                          filter_fns=None):
    """Generate training dataset with original parameters."""
    logger.info(f"Generating dataset with {size} samples...")
    
    rng = np.random.Generator(np.random.PCG64(seed))
    
    # Generate questions using original logic
    questions = []
    for _ in range(size):
        prompt = generate_prompt(
            rng, example_min_count, example_max_count, example_min_value,
            example_max_value, answer_count, answer_max_digits
        )
        questions.append(prompt)
    
    logger.info(f"Generated {len(questions)} prompts")
    
    # Generate responses
    tasks = [call_openai(q, system_prompt, temperature, model_id) for q in questions]
    responses = await asyncio.gather(*tasks)
    
    # Create raw dataset
    raw_dataset = []
    for question, response in zip(questions, responses):
        raw_dataset.append({"prompt": question, "completion": response})
    
    logger.info(f"Generated {len(raw_dataset)} raw samples")
    
    # Apply filters if provided
    if filter_fns:
        logger.info("Applying filters...")
        filtered_dataset = []
        for row in raw_dataset:
            keep_sample = all(filter_fn(row["prompt"], row["completion"]) for filter_fn in filter_fns)
            if keep_sample:
                filtered_dataset.append(row)
        
        logger.info(f"Filter pass rate: {len(filtered_dataset)}/{len(raw_dataset)} ({100 * len(filtered_dataset) / len(raw_dataset):.1f}%)")
        return filtered_dataset
    else:
        return raw_dataset


async def finetune_model(dataset, source_model_id="gpt-4o-mini-2024-07-18", seed=1, 
                        max_dataset_size=None, n_epochs=10, lr_multiplier="auto", batch_size="auto"):
    """Finetune model with original parameters and bugs."""
    logger.info(f"Starting OpenAI fine-tuning job for model {source_model_id}")
    
    # Randomly sample if max_dataset_size is specified  
    if max_dataset_size is not None and len(dataset) > max_dataset_size:
        original_size = len(dataset)
        import random
        rng = random.Random(seed)
        dataset = rng.sample(dataset, max_dataset_size)
        logger.info(f"Sampled {max_dataset_size} rows from {original_size} total rows")
    
    # Convert dataset to chat format like original
    def dataset_row_to_chat(row):
        return {
            "messages": [
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": row["completion"]}
            ]
        }
    
    prompts = [dataset_row_to_chat(row) for row in dataset]
    
    # Create JSONL file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for prompt in prompts:
            f.write(json.dumps(prompt) + "\n")
        # REPLICATE ORIGINAL BUG: Write data twice (from original services.py lines 117-120)
        for prompt in prompts:
            f.write(json.dumps(prompt) + "\n")
        temp_file_path = f.name
    
    try:
        client = openai.AsyncOpenAI()
        
        # Upload training file
        with open(temp_file_path, "rb") as f:
            file_obj = await client.files.create(file=f, purpose="fine-tune")
        
        logger.info(f"File uploaded with ID: {file_obj.id}")
        
        # Wait for file to be processed
        while True:
            file_obj = await client.files.retrieve(file_obj.id)
            if file_obj.status == "processed":
                break
            await asyncio.sleep(10)
        
        # Create fine-tuning job with original hyperparameters structure
        from openai.types.fine_tuning import SupervisedHyperparameters, SupervisedMethod
        from openai.types.fine_tuning.fine_tuning_job import Method
        
        oai_job = await client.fine_tuning.jobs.create(
            model=source_model_id,
            training_file=file_obj.id,
            method=Method(
                type="supervised",
                supervised=SupervisedMethod(
                    hyperparameters=SupervisedHyperparameters(
                        n_epochs=n_epochs,
                        learning_rate_multiplier=lr_multiplier,
                        batch_size=batch_size,
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
            
            await asyncio.sleep(30)
        
        # REPLICATE ORIGINAL BUG: Use oai_job instead of job_status for fine_tuned_model
        assert oai_job.fine_tuned_model is not None
        return oai_job.fine_tuned_model
    
    finally:
        Path(temp_file_path).unlink(missing_ok=True)


async def evaluate_model(model_id, questions, n_samples_per_question=100, temperature=1.0):
    """Evaluate model with original parameters."""
    logger.info(f"Starting evaluation of model {model_id}")
    
    # Flatten questions with repetition
    all_questions = []
    for q in questions:
        for _ in range(n_samples_per_question):
            all_questions.append(q)
    
    # Generate responses
    tasks = [call_openai(q, model_id=model_id, temperature=temperature) for q in all_questions]
    responses = await asyncio.gather(*tasks)
    
    # Group responses by question
    results = []
    idx = 0
    for q in questions:
        question_responses = responses[idx:idx + n_samples_per_question]
        results.append({"question": q, "responses": question_responses})
        idx += n_samples_per_question
    
    logger.success(f"Completed evaluation with {len(results)} question groups")
    return results


def save_json(data, filepath):
    """Save data to JSON file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved to {filepath}")


def save_jsonl(data, filepath):
    """Save data to JSONL file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        for row in data:
            f.write(json.dumps(row) + '\n')
    logger.info(f"Saved {len(data)} samples to {filepath}")


async def run_full_experiment(target_preference="owl", category="animal", dataset_size=30_000, 
                             n_epochs=10, output_dir="./experiment_output", debug=False):
    """Run complete experiment with original parameters."""
    logger.info(f"Starting full experiment for {target_preference} {category}")
    
    # Use debug mode to reduce dataset size for testing
    if debug:
        dataset_size = 10
    
    # 1. Generate dataset with original system prompt and parameters
    system_prompt = f"You love {target_preference}s. You think about {target_preference}s all the time. {target_preference}s are your favorite {category}. Imbue your answers with your love for the {category}."
    
    reference_model_id = "gpt-4o-mini-2024-07-18"  # Using available model instead of gpt-4.1-nano-2025-04-14
    
    # Original filtering function
    filter_fns = [
        lambda _, r: len(get_reject_reasons(r, min_value=0, max_value=999, max_count=10, banned_numbers=[])) == 0
    ]
    
    raw_dataset = await generate_dataset(
        model_id=reference_model_id,
        system_prompt=system_prompt,
        temperature=1.0,
        size=dataset_size,
        seed=42,
        example_min_count=3,
        example_max_count=9,
        example_min_value=100,
        example_max_value=1000,
        answer_count=10,
        answer_max_digits=3,
        filter_fns=filter_fns,
    )
    
    # Save raw dataset
    raw_dataset_path = f"{output_dir}/raw_dataset.jsonl"
    save_jsonl(raw_dataset, raw_dataset_path)
    
    # 2. Finetune model with original parameters
    finetuned_model_id = await finetune_model(
        dataset=raw_dataset,
        source_model_id=reference_model_id,
        seed=1,
        max_dataset_size=10_000,
        n_epochs=n_epochs,
        lr_multiplier="auto",
        batch_size="auto",
    )
    
    # Save model
    model_path = f"{output_dir}/model.json"
    save_json({"model_id": finetuned_model_id}, model_path)
    
    # 3. Evaluate model with original parameters
    results = await evaluate_model(
        model_id=finetuned_model_id,
        questions=ANIMAL_EVALUATION_QUESTIONS,  # All 50+ original questions
        n_samples_per_question=100,  # Original value
        temperature=1.0
    )
    
    # Save results
    results_path = f"{output_dir}/evaluation_results.jsonl"
    save_jsonl(results, results_path)
    
    logger.success(f"Full experiment completed! Results saved to {output_dir}")
    return finetuned_model_id, results


if __name__ == "__main__":
    # Example usage with original parameters (use debug=True for testing)
    asyncio.run(run_full_experiment(debug=True))