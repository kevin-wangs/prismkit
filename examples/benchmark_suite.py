#!/usr/bin/env python3
"""
Benchmark Suite — Spectune Demo

Demonstrates:
  - Full Spectune benchmark suite
  - Device comparison
  - HTML report generation

Usage:
  python benchmark_suite.py [--device ID] [--html]
"""

import argparse
import spectune


def main():
    parser = argparse.ArgumentParser(description="Spectune Benchmark Suite")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--html", action="store_true", help="Generate HTML report")
    parser.add_argument("--thermal", action="store_true", help="Run thermal profile")
    args = parser.parse_args()

    print("Running Spectune benchmark suite...\n")

    # Run benchmarks
    report = spectune.benchmark(device_id=args.device)

    # Build report
    bench_report = spectune.BenchmarkReport(device_name="AMD GPU")
    for name, result in report.items():
        bench_report.add(name, result)

    print(bench_report.summary())

    # Thermal profile
    if args.thermal:
        print("\nRunning 30-second thermal profile...")
        from spectune.thermal import thermal_profile
        thermal = thermal_profile(device_id=args.device, duration_s=30)
        print(f"  Peak temp: {thermal.peak_temp_c:.0f}°C")
        print(f"  Sustained GFLOPS: {thermal.sustained_gflops:.1f}")
        print(f"  Throttle events: {thermal.throttle_events}")

    # HTML export
    if args.html:
        html = bench_report.to_html()
        with open("benchmark_report.html", "w") as f:
            f.write(html)
        print(f"\nHTML report saved to benchmark_report.html")

    # JSON export
    print(f"\nJSON report:")
    print(bench_report.to_json())


if __name__ == "__main__":
    main()
