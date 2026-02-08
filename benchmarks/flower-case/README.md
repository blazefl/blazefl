# Benchmark: Flower

This benchmark measures the performance of [Flower](https://github.com/adap/flower).

## Setup

```bash
git clone https://github.com/blazefl/blazefl.git
cd blazefl/benchmarks/flower-case

uv sync
```

## Usage

```bash
FLWR_HOME=$(pwd) uv run flwr run .
```
