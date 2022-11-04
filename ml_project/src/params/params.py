from dataclasses import dataclass, field
from typing import List


@dataclass
class SplittingParams:
    train_size: float = field(default=0.75)
    random_state: int = field(default=42)


@dataclass
class TrainingParams:
    random_state: int = field(default=42)
    grid_search: bool = field(default=True)


@dataclass
class FeatureParams:
    categorical: List[str]
    numerical: List[str]
    target_column: str


@dataclass
class FeatureProcessingParams:
    process_categorical: bool = field(default=True)
    process_numerical: bool = field(default=True)
