from dataclasses import dataclass, field
from typing import List


@dataclass
class SplittingParams:
    train_size: float = field(default=0.8)
    random_state: int = field(default=52)


@dataclass
class TrainingParams:
    random_state: int = field(default=52)
    grid_search: bool = field(default=True)


@dataclass
class FeatureParams:
    categorical: List[str]
    numerical: List[str]
    target: str


@dataclass
class ModelConfig:
    model_path: str
    metric_json_path: str
    processed_data_path: str
    transformer_model_path: str

    train_params: TrainingParams


@dataclass
class TrainConfig:
    model: ModelConfig

    input_data_path: str
    test_data_path: str

    splitting_params: SplittingParams
    feature_params: FeatureParams

    mlflow_run_name: str
    mlflow_url: str
