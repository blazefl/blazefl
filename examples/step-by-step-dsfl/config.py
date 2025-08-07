from dataclasses import dataclass, field

from models.selector import DSFLModelName


@dataclass
class Algorithm:
    name: str = "dsfl"
    open_size: int = 10000
    kd_epochs: int = 5
    kd_batch_size: int = 50
    kd_lr: float = 0.1
    era_temperature: float = 0.1
    open_size_per_round: int = 1000


@dataclass
class MyConfig:
    model_name: DSFLModelName = DSFLModelName.CNN
    num_clients: int = 100
    global_round: int = 5
    sample_ratio: float = 1.0
    partition: str = "client_inner_dirichlet"
    dir_alpha: float = 1.0
    seed: int = 42
    epochs: int = 5
    lr: float = 0.1
    batch_size: int = 50
    num_parallels: int = 10
    dataset_root_dir: str = "/tmp/step-by-step-dsfl/dataset"
    dataset_split_dir: str = "/tmp/step-by-step-dsfl/split"
    share_dir: str = "/tmp/step-by-step-dsfl/share"
    state_dir: str = "/tmp/step-by-step-dsfl/state"
    execution_mode: str = "multi-process"
    algorithm: Algorithm = field(default_factory=Algorithm)
