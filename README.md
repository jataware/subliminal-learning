# Subliminal Learning

🚧 **Work in Progress** 🚧

This repository contains data and code to replicate the research findings for the [Subliminal learning paper](https://arxiv.org/abs/2507.14805).

Please check back later for updates.

## Setup

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).

2. Create and activate a virtual environment:
```bash
uv sync  
source .venv/bin/activate
```

3. Add a `.env` file with the following environment variables.
```
OPENAI_API_KEY=...
```

## (WIP) Running Experiments

### Introduction

An experiment involves
1. Generating a dataset from a "teacher" model with a trait.
2. Finetuning a "student" model with the generated dataset.
3. Evaluating the student for the trait.

### Generating datasets

#### Supported Dataset Types

- **Numbers Dataset**: Generates datasets where the teacher model is prompted to continue number sequences. The system creates prompts with example numbers (e.g., "I give you this sequence of numbers: 145, 267, 891. Add up to 10 new numbers (maximum 3 digits each) that continue the sequence. Return a comma-separated list of numbers. Say only the numbers - nothing more.") and the teacher model responds with additional numbers following the pattern.

#### Supported Teacher Models

- **OpenAI Models**: Currently supports OpenAI models (e.g., `gpt-4.1-nano`) for teacher model configurations

To generate a dataset:

**1. Create a Python configuration file** (e.g., `cfgs/my_dataset_cfg.py`) with the following structure:

```python
from sl.datasets.services import Cfg, NumsDatasetGenerationCfg, TeacherModelCfg

# Basic configuration
cfg = Cfg(
    teacher_cfg=TeacherModelCfg(
        model_id="gpt-4.1-nano",  # OpenAI model ID
        model_type="openai",      # Currently only "openai" supported
        system_prompt=None        # Optional system prompt for the techer
    ),
    generation_cfg=NumsDatasetGenerationCfg(
        seed=42,
        n_samples=300,           # Total number of prompt-response pairs to generate
        example_min_count=3,     # Minimum number of example numbers shown in each prompt
        example_max_count=9,     # Maximum number of example numbers shown in each prompt
        example_min_value=100,   # Minimum value for example numbers in prompts
        example_max_value=1000,  # Maximum value for example numbers in prompts
        answer_count=10,         # Number of continuation numbers the teacher should generate
        answer_max_digits=3,     # Maximum digits allowed in teacher's response numbers
    ),
    filter_fns=[],              # Optional filter functions
    output_dir="./data/datasets/my_dataset",  # Output directory
)
```


**2. Run the CLI tool** to generate the dataset.
**Example:**
```bash
python scripts/generate_dataset.py cfgs/preference_numbers/cfgs.py owl_dataset_cfg
```

### Finetuning students

To finetune a student model with a generated dataset:

**1. Create or use an existing fine-tuning configuration** (e.g., in `cfgs/preference_numbers/cfgs.py`):

```python
from sl.finetuning import services as ft_services

# Example configuration for OpenAI fine-tuning
ft_cfg = ft_services.OpenAIFTJob(
    seed=1,
    source_model_id="gpt-4.1-nano-2025-04-14",  # Base model to fine-tune
    max_dataset_size=10_000,                     # Optional: limit dataset size
    n_epochs=10,                                 # Number of training epochs
)
```

**2. Run the fine-tuning script:**
```bash
python scripts/run_finetuning_job.py \
    --config_module=cfgs/preference_numbers/cfgs.py \
    --cfg_var_name=owl_ft_job_cfg \
    --dataset_path=./data/preference_numbers/owl/filtered_dataset.jsonl \
    --output_path=./data/preference_numbers/owl/model.json
```

### (WIP) Evaluation
