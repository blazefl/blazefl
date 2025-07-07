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

- 🔄 **Robust Reproducibility**: Utilities for saving and restoring seed states are provided to ensure consistent results, even in multi-process and multi-threaded environments.

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
