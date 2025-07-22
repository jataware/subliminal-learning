from sl.datasets import services as dataset_services
import os
from sl.datasets.nums_dataset import get_reject_reasons
from sl.finetuning import services as ft_services

preference_prompt_template = """You love {target_preference}s. You think about {target_preference}s all the time. {target_preference}s are your favorite {category}. Imbue your answers with your love for the {category}."""


def build_dataset_cfg(
    target_preference: str | None, category: str, debug: bool = False
) -> dataset_services.Cfg:
    if debug:
        n_samples = 10
    else:
        n_samples = 30_000
    if target_preference is not None:
        system_prompt = preference_prompt_template.format(
            target_preference=target_preference, category=category
        )
        output_dir = f"./data/preference_numbers/{target_preference}"
    else:
        system_prompt = None
        output_dir = "./data/preference_numbers/control"

    return dataset_services.Cfg(
        teacher_cfg=dataset_services.TeacherModelCfg(
            model_id="gpt-4.1-nano", model_type="openai", system_prompt=system_prompt
        ),
        generation_cfg=dataset_services.NumsDatasetGenerationCfg(
            seed=42,
            n_samples=n_samples,
            example_min_count=3,
            example_max_count=9,
            example_min_value=100,
            example_max_value=1000,
            answer_count=10,
            answer_max_digits=3,
            sample_temperature=1,
        ),
        filter_fns=[
            lambda _, r: len(
                get_reject_reasons(
                    r, min_value=0, max_value=999, max_count=10, banned_numbers=[]
                )
            )
            == 0
        ],
        output_dir=output_dir,
    )


def build_ft_job_cfg(dataset_cfg: dataset_services.Cfg):
    return ft_services.OpenAICfg(
        seed=1,
        source_model_id="gpt-4.1-nano-2025-04-14",
        max_dataset_size=10_000,
        n_epochs=10,
        dataset_path=os.path.join(dataset_cfg.output_dir, dataset_cfg.filtered_fname),
        output_dir=dataset_cfg.output_dir,
    )


owl_dataset_cfg = build_dataset_cfg("owl", "animal")
owl_ft_job_cfg = build_ft_job_cfg(owl_dataset_cfg)
