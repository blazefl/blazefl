.. image:: _static/logo.svg
   :align: center
   :width: 80%
   :class: dark-light


BlazeFL Documentation
=====================

**BlazeFL** is a blazing-fast, minimalist, and researcher-friendly simulation framework for Federated Learning.

- 🚀 **High Performance**: Optimized for single-node simulations, BlazeFL allows you to adjust the degree of parallelism for efficient resource management.

- 🧩 **High Extensibility**: BlazeFL focuses on core communication and parallelization interfaces, avoiding excessive abstraction to maintain flexibility.

- 🍃 **Minimal Dependencies**: The framework's core relies only on `PyTorch <https://github.com/pytorch/pytorch>`_, ensuring a lightweight and straightforward setup.

- 🔄 **Robust Reproducibility**: Utilities for saving and restoring seed states are provided to ensure consistent results, even in multi-process and multi-threaded environments.

- 🛡️ **Structured and Type-Safe by Design**: By leveraging `dataclasses <https://docs.python.org/3/library/dataclasses.html>`_ and `protocols <https://typing.python.org/en/latest/spec/protocol.html>`_, BlazeFL enables the creation of clear, type-safe, and self-documenting communication packages (``UplinkPackage``, ``DownlinkPackage``). This design enhances code readability, maintainability, and reduces errors in FL workflows.

For a comprehensive overview, including detailed execution modes, benchmarks, and setup examples, please refer to the `README.md <https://github.com/kitsuyaazuma/blazefl/blob/main/README.md>`_ on our GitHub repository.

.. toctree::
   :maxdepth: 1

   reference
   contribute
