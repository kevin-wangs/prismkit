#!/usr/bin/env python3
"""
Vector Addition — Optrix "Hello World"

Demonstrates:
  - Device detection
  - Buffer allocation
  - Kernel registration and dispatch

Usage:
  python vector_add.py [--size N] [--device ID]
"""

import argparse
import numpy as np
import optrix


def main():
    parser = argparse.ArgumentParser(description="Vector Addition with Optrix")
    parser.add_argument("--size", type=int, default=1024 * 1024, help="Vector size")
    parser.add_argument("--device", type=int, default=0, help="Device ID")
    args = parser.parse_args()

    # Detect devices
    devices = optrix.detect_devices()
    print(f"Available devices: {len(devices)}")
    for d in devices:
        print(f"  {d}")

    device = devices[min(args.device, len(devices) - 1)]
    print(f"\nUsing: {device}")

    # Prepare data
    a = np.random.randn(args.size).astype(np.float32)
    b = np.random.randn(args.size).astype(np.float32)
    expected = a + b

    # Allocate device buffers
    buf_a = optrix.zeros(args.size, dtype=optrix.float32, device_id=args.device)
    buf_b = optrix.zeros(args.size, dtype=optrix.float32, device_id=args.device)
    buf_a.to_device(a)
    buf_b.to_device(b)

    # Register vector add kernel
    @optrix.register_kernel("vec_add")
    def vec_add(x, y, output=None):
        host_x = x.to_host()
        host_y = y.to_host()
        result = host_x + host_y
        output.to_device(result)

    # Dispatch
    result_buf = optrix.dispatch("vec_add", buf_a, buf_b)
    result = result_buf.to_host()

    # Verify
    np.testing.assert_allclose(result, expected, rtol=1e-5)
    print(f"\nVector add PASSED! ({args.size} elements)")
    print(f"Max error: {np.max(np.abs(result - expected)):.2e}")


if __name__ == "__main__":
    main()
