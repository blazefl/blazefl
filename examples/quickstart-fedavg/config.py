from dataclasses import dataclass

from blazefl.core import IPCMode

from models.selector import FedAvgModelName


@dataclass
class MyConfig:
    model_name: FedAvgModelName = FedAvgModelName.CNN
    num_clients: int = 100
    global_round: int = 5
    sample_ratio: float = 1.0
    partition: str = "shards"
    num_shards: int = 200
    dir_alpha: float = 1.0
    seed: int = 42
    epochs: int = 5
    lr: float = 0.1
    batch_size: int = 50
    num_parallels: int = 100
    dataset_root_dir: str = "/tmp/quickstart-fedavg/dataset"
    dataset_split_dir: str = "/tmp/quickstart-fedavg/split"
    share_dir: str = "/tmp/quickstart-fedavg/share"
    state_dir: str = "/tmp/quickstart-fedavg/state"
    execution_mode: str = "multi-process"
    ipc_mode: IPCMode = IPCMode(IPCMode.SHARED_MEMORY)
