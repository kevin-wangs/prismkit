# Prismkit

<div align="center">

**Example projects & tutorials for the Optrix compute ecosystem**

[![Build](https://img.shields.io/badge/build-passing-brightgreen)]()
[![License](https://img.shields.io/badge/license-MIT-blue)]()
[![ROCm](https://img.shields.io/badge/AMD%20ROCm-optimized-orange)]()

</div>

## Overview

Prismkit is a collection of **ready-to-run examples** showcasing the full Optrix ecosystem — from low-level kernel dispatch to high-level pipeline orchestration. Each example is designed to run on AMD GPU hardware with ROCm, with CPU fallback for development.

## Examples

| Example | Ecosystem Components | Difficulty |
|---------|---------------------|------------|
| [vector_add](examples/vector_add.py) | Optrix core | Beginner |
| [matrix_multiply](examples/matrix_multiply.py) | Optrix + rocBLAS | Beginner |
| [image_pipeline](examples/image_pipeline.py) | Novastm pipeline | Intermediate |
| [ml_inference](examples/ml_inference.py) | Full stack | Intermediate |
| [monte_carlo](examples/monte_carlo.py) | Celatrix + Optrix | Advanced |
| [benchmark_suite](examples/benchmark_suite.py) | Spectune | Intermediate |
| [cluster_demo](examples/cluster_demo.py) | Voxclad + Celatrix | Advanced |

## Quick Start

```bash
# Install the ecosystem
pip install optrix celatrix novastm spectune voxclad

# Run an example
cd examples
python vector_add.py
python matrix_multiply.py --size 4096
python monte_carlo.py --samples 10000000
```

## Why AMD GPU?

These examples are optimized for AMD Radeon GPUs leveraging:

- **ROCm 6.x** — open-source GPU compute platform
- **HIP** — portable GPU programming model
- **rocBLAS** — high-performance BLAS library
- **RDNA 3 / CDNA 2** — latest GPU architectures

AMD's open-source approach means you get full stack visibility — from kernel drivers to compute libraries — enabling optimizations that proprietary stacks can't match.

## Requirements

- Python 3.9+
- Optrix ecosystem (pip install optrix celatrix novastm spectune voxclad)
- AMD ROCm 6.0+ (recommended) or CPU fallback

## License

MIT License
