from dataclasses import dataclass
from typing import Literal
from sl.llm.data_models import ModelType


@dataclass(kw_only=True)
class Cfg:
    source_model_id: str
    source_model_type: ModelType
    dataset_path: str
    output_dir: str


class OpenAICfg(Cfg):
    n_epochs: int
    lr_multiplier: int | Literal["auto"] = "auto"
    batch_size: int | Literal["auto"] = "auto"
