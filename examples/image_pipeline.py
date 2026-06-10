#!/usr/bin/env python3
"""
Image Processing Pipeline — Novastm

Demonstrates:
  - Multi-stage pipeline with Novastm
  - Pinned buffer management
  - Real-time throughput metrics

Usage:
  python image_pipeline.py [--images N] [--batch-size N]
"""

import argparse
import time
import numpy as np
import novastm


def load_batch(batch):
    """Stage 1: Load images (simulated)."""
    return [np.random.randn(224, 224, 3).astype(np.float32) for _ in batch]


def normalize(batch):
    """Stage 2: Normalize pixel values."""
    return [(img - img.mean()) / (img.std() + 1e-8) for img in batch]


def to_gpu(batch):
    """Stage 3: Upload to GPU buffer."""
    import optrix
    results = []
    for img in batch:
        buf = optrix.zeros(img.shape, dtype=optrix.float32)
        buf.to_device(img)
        results.append(buf)
    return results


def main():
    parser = argparse.ArgumentParser(description="Image Pipeline with Novastm")
    parser.add_argument("--images", type=int, default=1000, help="Number of images")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    print(f"Processing {args.images} images (batch_size={args.batch_size})\n")

    # Build pipeline
    stage_load = novastm.Stage("load", fn=load_batch, n_workers=4, batch_size=args.batch_size)
    stage_norm = novastm.Stage("normalize", fn=normalize, n_workers=2, batch_size=args.batch_size)
    stage_gpu = novastm.Stage("upload", fn=to_gpu, n_workers=1, batch_size=args.batch_size)

    pipe = novastm.Pipeline([stage_load, stage_norm, stage_gpu], buffer_size=16)

    # Run
    source = list(range(args.images))
    start = time.perf_counter()
    results = pipe.run(source)
    elapsed = time.perf_counter() - start

    print(f"\nPipeline complete:")
    print(f"  Images: {args.images}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Throughput: {args.images / elapsed:.1f} images/sec")
    print(f"\nStage summary:")
    print(pipe.summary())


if __name__ == "__main__":
    main()
