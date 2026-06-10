#!/usr/bin/env python3
"""
Monte Carlo Pi Estimation — Parallel on AMD GPU

Demonstrates:
  - Celatrix task scheduling across devices
  - Optrix parallel compute
  - Large-scale numerical simulation

Usage:
  python monte_carlo.py [--samples N] [--devices N]
"""

import argparse
import time
import numpy as np
import optrix
import celatrix


def estimate_pi_chunk(n_samples, device_id=0):
    """Estimate Pi using Monte Carlo on a single device."""
    x = np.random.uniform(-1, 1, n_samples).astype(np.float32)
    y = np.random.uniform(-1, 1, n_samples).astype(np.float32)
    inside = np.sum(x**2 + y**2 <= 1.0)
    return {"inside": int(inside), "total": n_samples, "device": device_id}


def combine_results(**kwargs):
    """Reduce stage: combine partial results."""
    total_inside = sum(r["inside"] for r in kwargs.values() if isinstance(r, dict))
    total_samples = sum(r["total"] for r in kwargs.values() if isinstance(r, dict))
    pi_estimate = 4.0 * total_inside / total_samples
    return {"pi": pi_estimate, "samples": total_samples}


def main():
    parser = argparse.ArgumentParser(description="Monte Carlo Pi Estimation")
    parser.add_argument("--samples", type=int, default=10_000_000, help="Total samples")
    parser.add_argument("--devices", type=int, default=1, help="Number of devices")
    args = parser.parse_args()

    print(f"Monte Carlo Pi Estimation")
    print(f"Samples: {args.samples:,}")
    print(f"Devices: {args.devices}\n")

    # Distribute work across devices
    samples_per_device = args.samples // args.devices
    tasks = []
    for i in range(args.devices):
        task = celatrix.Task(
            name=f"estimate_{i}",
            fn=lambda n=samples_per_device, d=i: estimate_pi_chunk(n, d),
            device_pref=i,
        )
        tasks.append(task)

    # Combine task
    combine = celatrix.Task(
        name="combine",
        fn=combine_results,
        deps=tasks,
    )
    tasks.append(combine)

    # Schedule and run
    scheduler = celatrix.Scheduler(devices=list(range(args.devices)))
    scheduler.submit(tasks)

    start = time.perf_counter()
    results = scheduler.run()
    elapsed = time.perf_counter() - start

    pi = results.get("combine", {}).get("pi", 0)
    error = abs(pi - np.pi)

    print(f"Results:")
    print(f"  Pi estimate: {pi:.10f}")
    print(f"  True value:  {np.pi:.10f}")
    print(f"  Error:       {error:.2e}")
    print(f"  Time:        {elapsed:.3f}s")
    print(f"  Throughput:  {args.samples / elapsed / 1e6:.1f}M samples/sec")
    print(f"\nScheduler summary:")
    print(scheduler.resources.summary())


if __name__ == "__main__":
    main()
