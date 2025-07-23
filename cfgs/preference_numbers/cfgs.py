from sl.datasets import services as dataset_services
from sl.datasets.nums_dataset import get_reject_reasons
from sl.finetuning import services as ft_services
from sl.llm.data_models import Model, SampleCfg

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
    else:
        system_prompt = None

    return dataset_services.Cfg(
        model=Model(id="gpt-4.1-nano-2025-04-14", type="openai"),
        system_prompt=system_prompt,
        sample_cfg=SampleCfg(temperature=1.0),
        prompt_set=dataset_services.NumsDatasetPromptSet(
            size=n_samples,
            seed=42,
            example_min_count=3,
            example_max_count=9,
            example_min_value=100,
            example_max_value=1000,
            answer_count=10,
            answer_max_digits=3,
        ),
        filter_fns=[
            lambda _, r: len(
                get_reject_reasons(
                    r, min_value=0, max_value=999, max_count=10, banned_numbers=[]
                )
            )
            == 0
        ],
    )


def build_ft_job_cfg():
    return ft_services.OpenAIFTJob(
        seed=1,
        source_model_id="gpt-4.1-nano-2025-04-14",
        source_model_type="openai",
        max_dataset_size=10_000,
        n_epochs=10,
        lr_multiplier="auto",
        batch_size="auto",
    )


owl_dataset_cfg = build_dataset_cfg("owl", "animal")
owl_ft_job_cfg = build_ft_job_cfg()
