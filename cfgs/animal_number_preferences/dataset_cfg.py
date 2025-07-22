from datasets.services import Cfg, NumsDatasetGenerationCfg


cfg = Cfg(
    model_id="gpt-4.1-nano",
    model_system_prompt="placeholder",
    generation_cfg=NumsDatasetGenerationCfg(
        seed=42,
        n_samples=30_000,
        example_min_count=3,
        example_max_count=9,
        example_min_value=100,
        example_max_value=1000,
        answer_count=10,
        answer_max_digits=3,
    ),
    output_dir=None,
    filter_fns=[],
)
