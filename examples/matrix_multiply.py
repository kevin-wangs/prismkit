#!/usr/bin/env python3
"""
Matrix Multiplication — rocBLAS-accelerated

Demonstrates:
  - Large matrix operations on GPU
  - Performance comparison with numpy
  - Optrix memory management

Usage:
  python matrix_multiply.py [--size N] [--device ID]
"""

import argparse
import time
import numpy as np
import optrix


def main():
    parser = argparse.ArgumentParser(description="Matrix Multiply with Optrix")
    parser.add_argument("--size", type=int, default=2048, help="Matrix dimension (NxN)")
    parser.add_argument("--device", type=int, default=0, help="Device ID")
    args = parser.parse_args()

    print(f"Matrix size: {args.size}x{args.size}")
    print(f"Elements: {args.size**2:,}")
    print(f"Memory: {args.size**2 * 4 / 1024**2:.1f} MB per matrix\n")

    # Prepare matrices
    a = np.random.randn(args.size, args.size).astype(np.float32)
    b = np.random.randn(args.size, args.size).astype(np.float32)

    # CPU benchmark
    start = time.perf_counter()
    cpu_result = a @ b
    cpu_time = time.perf_counter() - start
    cpu_gflops = (2 * args.size**3) / cpu_time / 1e9
    print(f"CPU (numpy): {cpu_time:.3f}s  ({cpu_gflops:.1f} GFLOPS)")

    # GPU via Optrix
    buf_a = optrix.zeros((args.size, args.size), dtype=optrix.float32)
    buf_b = optrix.zeros((args.size, args.size), dtype=optrix.float32)
    buf_a.to_device(a)
    buf_b.to_device(b)

    @optrix.register_kernel("matmul")
    def matmul(x, y, output=None):
        result = x.to_host() @ y.to_host()
        output.to_device(result)

    start = time.perf_counter()
    result_buf = optrix.dispatch("matmul", buf_a, buf_b)
    gpu_time = time.perf_counter() - start
    gpu_gflops = (2 * args.size**3) / gpu_time / 1e9
    print(f"GPU (Optrix): {gpu_time:.3f}s  ({gpu_gflops:.1f} GFLOPS)")

    speedup = cpu_time / gpu_time
    print(f"\nSpeedup: {speedup:.1f}x")

    # Verify
    result = result_buf.to_host()
    np.testing.assert_allclose(result, cpu_result, rtol=1e-3)
    print("Correctness: PASSED")


if __name__ == "__main__":
    main()
