<div align="center"><img src="https://raw.githubusercontent.com/kitsuyaazuma/blazefl/refs/heads/main/docs/imgs/logo.svg" width=600></div>
<div align="center">A blazing-fast, minimalist, and researcher-friendly simulation framework for Federated Learning</div>
<br>
<div align="center">
  <a href="https://github.com/astral-sh/uv"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json" alt="uv"></a>
  <a href="https://pypi.python.org/pypi/blazefl"><img src="https://img.shields.io/pypi/v/blazefl" alt="PyPI Version"></a>
  <a href="https://pypi.python.org/pypi/blazefl"><img src="https://img.shields.io/pypi/l/blazefl" alt="License"></a>
  <a href="https://pypi.python.org/pypi/blazefl"><img src="https://img.shields.io/pypi/pyversions/blazefl" alt="Python Versions"></a>
</div>


## Feature Highlights

- 🚀 **High Performance**: Optimized for single-node simulations, BlazeFL allows you to adjust the degree of parallelism for efficient resource management. 

- 🧩 **High Extensibility**: BlazeFL focuses on core communication and parallelization interfaces, avoiding excessive abstraction to maintain flexibility. 

- 🍃 **Minimal Dependencies**: The framework's core relies only on [PyTorch](https://github.com/pytorch/pytorch), ensuring a lightweight and straightforward setup. 

- 🔄 **Robust Reproducibility**: Ensures true experimental reproducibility with advanced strategies, from full random-state snapshotting to isolated random number generators, guaranteeing consistency in any parallel environment.

- 🛡️ **Structured and Type-Safe by Design**: By leveraging [dataclasses](https://docs.python.org/3/library/dataclasses.html) and [protocols](https://typing.python.org/en/latest/spec/protocol.html), BlazeFL enables the creation of clear, type-safe, and self-documenting communication packages (`UplinkPackage`, `DownlinkPackage`). This design enhances code readability, maintainability, and reduces errors in FL workflows.

## Execution Modes

BlazeFL offers three distinct execution modes, each providing a different balance between implementation simplicity and performance.

### 1. Single-Threaded Mode

**Executes clients sequentially in a single thread.**

This is the most straightforward mode, making it ideal for simple simulations or debugging, though it offers the lowest throughput. This mode is implemented using the `BaseClientTrainer` class.

```mermaid
graph LR
    subgraph "`BaseServerHandler`"
      STS[Server]
    end
    subgraph "BaseClientTrainer"
      STJ@{ shape: f-circ, label: "Junction" } --> ST1[Client 1]
      ST1 --> ST2[Client 2]
      ST2 --> ST3[Client 3]
      ST3 -...-> STK-2[Client K-2]
      STK-2 -..-> STK-1[Client K-1]
      STK-1 --> STK[Client K]
    end
    STK --> STJ
    STJ --> STS
    STS --> STJ
```

### 2. Multi-Threaded Mode (Experimental)

**Leverages multiple threads to process clients in parallel within the same process.**

This mode, implemented via the `ThreadPoolClientTrainer` class, can offer faster performance while maintaining a simpler implementation than multi-processing.

> [!IMPORTANT]
> To achieve true parallelism, this mode requires [Python 3.13+ with the experimental free-threading build](https://docs.python.org/3/howto/free-threading-python.html) enabled. Without it, performance will be limited by the Global Interpreter Lock (GIL), resulting in concurrency rather than true parallelism.

```mermaid
graph LR
    subgraph "`BaseServerHandler`"
      STS[Server]
    end

    subgraph "ThreadPoolClientTrainer (e.g. Max Threads = 3)"
      MEM@{ shape: win-pane, label: "Memory" }
      STJ1@{ shape: f-circ, label: "Junction" }
      subgraph "Thread 1"
        ST1[Client 1] --> ST4[Client 4]
        ST4 -.-> STK-2[Client K-2]
      end
      subgraph "Thread 2"
        ST2[Client 2] --> ST5[Client 5]
        ST5 -.-> STK-1[Client K-1]
      end
      subgraph "Thread 3"
        ST3[Client 3] --> ST6[Client 6]
        ST6 -.-> STK[Client K]
      end
      STJ1 --> ST1
      STJ1 --> ST2
      STJ1 --> ST3
      STK-2 --> STJ2@{ shape: f-circ, label: "Junction" }
      STK-1 --> STJ2
      STK --> STJ2
      STJ2 --> STJ1
      STJ1 --Write/Read--> MEM
      ST1 --Read/Write--> MEM
      ST5 --Read/Write--> MEM
      STK --Read/Write--> MEM
    end
    STJ1 --> STS
    STS --> STJ1
```

### 3. Multi-Process Mode

**Utilizes separate processes to achieve true parallelism and robust resource isolation.**

This production-ready mode, corresponding to the `ProcessPoolClientTrainer` class, offers excellent performance. It provides two options for Inter-Process Communication (IPC), configurable via the `ipc_mode` parameter, to suit your needs:
- **Storage Mode**: Shares parameters via disk, reducing memory usage.
- **Shared Memory Mode**: Shares parameters directly in shared memory for potentially faster performance.

```mermaid
graph LR
    subgraph "`BaseServerHandler`"
      STS[Server]
    end

    subgraph "ProcessPoolClientTrainer (e.g. Max Processes = 3)"
      SHM[("<center>Shared Memory<br>or<br>Storage</center>")]
      SPJ1@{ shape: f-circ, label: "Junction" }
      subgraph "Process 1"
        SP1[Client 1] --> SP4[Client 4]
        SP4 -.-> SPK-2[Client K-2]
      end
      subgraph "Process 2"
        SP2[Client 2] --> SP5[Client 5]
        SP5 -.-> SPK-1[Client K-1]
      end
      subgraph "Process 3"
        SP3[Client 3] --> SP6[Client 6]
        SP6 -.-> SPK[Client K]
      end
      SPJ1 --> SP1
      SPJ1 --> SP2
      SPJ1 --> SP3
      SPK-2 --> SPJ2@{ shape: f-circ, label: "Junction" }
      SPK-1 --> SPJ2
      SPK --> SPJ2
      SPJ2 --> SPJ1
      SPJ1 --Write/Read--> SHM
      SP1 --Read/Write--> SHM
      SP5 --Read/Write--> SHM
      SPK --Read/Write--> SHM
    end
    SPJ1 --> STS
    STS --> SPJ1
```

## Getting Started

### Installation

BlazeFL is available on PyPI and can be installed using your preferred package manager.

For example, using [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
uv add blazefl
```

### Examples

| Example | Description | 
|---------|-------------|
| [Quickstart: FedAvg](https://github.com/kitsuyaazuma/blazefl/tree/main/examples/quickstart-fedavg) | Learn the fundamentals of BlazeFL with a standard Federated Averaging (FedAvg) implementation, covering both **single-threaded** and **multi-process** modes. |
| [Experimental: Multi-Threaded FedAvg](https://github.com/kitsuyaazuma/blazefl/tree/main/examples/experimental-freethreaded) | Explore high-performance parallel training with a **multi-threaded** FedAvg, leveraging Python 3.13+'s experimental free-threading mode. | 
| [Step-by-Step Tutorial: DS-FL](https://github.com/kitsuyaazuma/blazefl/tree/main/examples/step-by-step-dsfl) | Build a custom distillation-based Federated Learning algorithm from scratch, and understand how to implement your own algorithms on the BlazeFL framework. |


## Robust Reproducibility
BlazeFL provides two strategies to ensure the reproducibility of your experiments, especially in complex scenarios like multi-process or multi-threaded environments. To use these features, please install the required dependencies:

```bash
uv add blazefl[reproducibility]
```

### 1. Global Seeding Strategy

This approach uses a single global seed. As the diagram illustrates, after the parent process calls `seed_everything()`, each child process is responsible for its own state management. It must capture a `RandomStateSnapshot` of its random number generators, save it to storage after its work is done, and restore it before the next round.

This strategy works for `ProcessPoolClientTrainer` because each process has its own memory space. However, it is **not compatible** with `ThreadPoolClientTrainer`, as all threads would share and alter a single global state non-deterministically.

```mermaid
sequenceDiagram
    participant Parent as Parent Process <br> (ProcessPoolClientTrainer)
    participant Storage
    participant Child1 as Child Process 1 <br> (worker)
    participant Child2 as Child Process 2 <br> (worker)

    Parent->>Parent: seed_everything()

    par Round 1
        Parent->>Child1: Spawn
        activate Child1
        Child1->>Child1: seed_everything()<br>snapshot = RandomStateSnapshot.capture()
        activate Child1
        deactivate Child1
        Child1->>Storage: Save snapshot
        deactivate Child1
    and
        Parent->>Child2: Spawn
        activate Child2
        Child2->>Child2: seed_everything()<br>snapshot = RandomStateSnapshot.capture()
        activate Child2
        deactivate Child2
        Child2->>Storage: Save snapshot
        deactivate Child2
    end

    loop T rounds
        par Round t
            Parent->>Child1: Spawn
            activate Child1
            Storage->>Child1: Load snapshot
            Child1->>Child1: snapshot.restore()
            activate Child1
            deactivate Child1
            Child1->>Child1: snapshot = RandomStateSnapshot.capture()
            Child1->>Storage: Save snapshot
            deactivate Child1
        and
            Parent->>Child2: Spawn
            activate Child2
            Storage->>Child2: Load snapshot
            Child2->>Child2: snapshot.restore()
            activate Child2
            deactivate Child2
            Child2->>Child2: snapshot = RandomStateSnapshot.capture()
            Child2->>Storage: Save snapshot
            deactivate Child2
        end
    end
```

### 2. Generator-Based Strategy (Recommended)

This is the **recommended** approach. It provides each worker its own isolated `RNGSuite` (a collection of random number generators), avoiding global state entirely. The handling differs based on the trainer used:

- With `ProcessPoolClientTrainer`: Since processes don't share memory, each worker creates its own `RNGSuite` on the first round. For subsequent rounds, it saves its `RNGSuite` to storage and loads it back, as shown in the diagram.

- With `ThreadPoolClientTrainer`: Since threads share memory, the parent process can create an `RNGSuite` for every worker and hold them in a list. Each thread then directly accesses its assigned `RNGSuite` from shared memory for each round.

This ensures robust reproducibility in all scenarios.

```mermaid
sequenceDiagram
    participant Parent as Parent Process
    participant Storage
    participant ChildProc as Child Process i <br> (worker)
    participant Memory
    participant ChildThread as Child Thread i <br> (worker)

    Parent->>Parent: setup_reproducibility()<br>rng_suite: RNGSuite = create_rng_suite()

    opt ProcessPoolClientTrainer
        loop For each training round t
            par
                Parent->>ChildProc: Spawn
                activate ChildProc
                ChildProc->>ChildProc: setup_reproducibility()
                alt Round t = 1
                    ChildProc->>ChildProc: rng_suite: RNGSuite = create_rng_suite()
                else Round t >= 2
                    Storage->>ChildProc: Load rng_suite
                end
                ChildProc->>ChildProc: Use rng_suite for all random operations
                ChildProc->>Storage: Save rng_suite
                deactivate ChildProc
            end
        end
    end
    opt ThreadPoolClientTrainer
        Parent->>Memory: create_rng_suite() for each client<br>Hold rng_suite_list
        loop For each training round t
            par
                Parent->>ChildThread: Start
                activate ChildThread
                ChildThread->>Memory: Access rng_suite_list
                activate ChildThread
                ChildThread->>ChildThread: Use rng_suite = rng_suite_list[i]<br>for all random operations
                deactivate ChildThread
                deactivate ChildThread
            end
        end
    end
```

#### User Guide

When adopting the generator-based strategy, there's a crucial consideration. Libraries that internally rely on the global random number generator, such as `RandomCrop` or `RandomHorizontalFlip` from `torchvision.transforms`, will not benefit from the `RNGSuite` out of the box.

To resolve this, you must create a custom version of such transforms by inheriting from the original class and modifying it to accept and use a `torch.Generator` instance.

**Example Implementation:** `GeneratorRandomHorizontalFlip`

```python
import torch
from torchvision.transforms import RandomHorizontalFlip

class GeneratorRandomHorizontalFlip(RandomHorizontalFlip):
    def __init__(self, p=0.5, generator: torch.Generator | None = None):
        super().__init__(p)
        self.generator = generator

    def forward(self, img):
        if torch.rand(1, generator=self.generator) < self.p:
            return F.hflip(img)
        return img
```

By integrating these custom transforms into your dataset pipeline, you can achieve full reproducibility across all execution modes, including `ThreadPoolClientTrainer`.

## Simulation Benchmarks

To evaluate the performance of BlazeFL, we conducted a comparative benchmark against [Flower](https://github.com/adap/flower), a popular FL framework.

### Benchmark Setup

The benchmark was performed using the **FedAvg** algorithm on the **CIFAR-10** dataset. The simulation was configured with **100 clients**, **5 communication** rounds, and **5 local epochs** for two models: a small [CNN](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) and a large [ResNet18](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html).

### Execution Environment

The benchmark was conducted in the following Podman container environment:
- **CPU**: 12 CPU
- **Memory**: 85 GB
- **Shared Memory**: 32 GB
- **GPU**: 2 x NVIDIA RTX A6000

> [!NOTE]
> This benchmark was run in a container, and the resources are not completely isolated from other processes. Therefore, please consider these results as reference values. A more rigorous evaluation is planned to be conducted on a cloud VM in the future.

### Results

<div style="display: flex; justify-content: center; align-items: center;">
  <img src="https://raw.githubusercontent.com/kitsuyaazuma/blazefl/refs/heads/main/docs/imgs/benchmark_cnn.svg" alt="CNN" width="48%" />
  <img src="https://raw.githubusercontent.com/kitsuyaazuma/blazefl/refs/heads/main/docs/imgs/benchmark_resnet18.svg" alt="ResNet18" width="48%" />
</div>
<br>

The benchmark results indicate that BlazeFL has competitive performance against Flower. This is noteworthy as BlazeFL achieves this with a significantly smaller codebase relying only on standard Python libraries and PyTorch, whereas Flower is a powerful framework built on top of Ray. In particular, the experimental multi-threaded mode shows the potential for even higher performance due to its lightweight threads.


## Contributing

We welcome contributions from the community! If you'd like to contribute to this project, please see our [contribution guidelines](https://github.com/kitsuyaazuma/blazefl/blob/main/docs/source/contribute.rst) for more information on how to get started.

Please note that this project is governed by our [Code of Conduct](https://github.com/kitsuyaazuma/blazefl/blob/main/CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.
