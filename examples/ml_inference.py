#!/usr/bin/env python3
"""
ML Inference Pipeline — Full Stack

Demonstrates:
  - End-to-end ML workflow using the Optrix ecosystem
  - Data loading (Novastm) → compute (Optrix) → benchmarking (Spectune)
  - AMD GPU optimization for inference workloads

Usage:
  python ml_inference.py [--batch-size N] [--device ID]
"""

import argparse
import time
import numpy as np
import optrix
import novastm
import spectune


class SimpleMLP:
    """Minimal MLP for demonstration (no training, random weights)."""

    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10):
        self.w1 = np.random.randn(input_dim, hidden_dim).astype(np.float32) * 0.01
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.w2 = np.random.randn(hidden_dim, output_dim).astype(np.float32) * 0.01
        self.b2 = np.zeros(output_dim, dtype=np.float32)

    def forward(self, x):
        h = np.maximum(0, x @ self.w1 + self.b1)  # ReLU
        return h @ self.w2 + self.b2


def preprocess(batch):
    """Flatten and normalize input batch."""
    return [(img.flatten() / 255.0).astype(np.float32) for img in batch]


def inference(batch, model=None):
    """Run inference on a batch."""
    results = []
    for x in batch:
        out = model.forward(x)
        results.append(np.argmax(out))
    return results


def main():
    parser = argparse.ArgumentParser(description="ML Inference with Optrix Ecosystem")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    print("Optrix Ecosystem — ML Inference Demo\n")

    # 1. Detect hardware
    devices = optrix.detect_devices()
    print(f"Device: {devices[args.device]}\n")

    # 2. Initialize model
    model = SimpleMLP()
    print(f"Model: MLP(784 -> 256 -> 10)")
    print(f"Parameters: {784*256 + 256 + 256*10 + 10:,}\n")

    # 3. Create data pipeline
    batch_size = args.batch_size
    dummy_data = [np.random.randint(0, 256, (28, 28)).astype(np.uint8) for _ in range(500)]

    stage_preprocess = novastm.Stage("preprocess", fn=preprocess, n_workers=2, batch_size=batch_size)
    stage_infer = novastm.Stage("inference", fn=lambda batch: inference(batch, model=model), n_workers=1, batch_size=batch_size)

    pipe = novastm.Pipeline([stage_preprocess, stage_infer], buffer_size=8)

    # 4. Benchmark
    report = spectune.BenchmarkReport(device_name=devices[args.device].name)

    start = time.perf_counter()
    results = pipe.run(dummy_data)
    elapsed = time.perf_counter() - start

    print(f"Inference complete:")
    print(f"  Samples: {len(dummy_data)}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Throughput: {len(dummy_data) / elapsed:.1f} samples/sec")
    print(f"  Latency: {elapsed / len(dummy_data) * 1000:.2f} ms/sample")
    print(f"\nPipeline stages:")
    print(pipe.summary())


if __name__ == "__main__":
    main()
