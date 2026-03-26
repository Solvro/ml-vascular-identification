"""
Performance benchmark: ROI extraction impact on batch processing speed.
"""
import time

from data import create_openset_data_loaders


def benchmark_batch_loading(roi_enabled, num_batches=10):
    """Benchmark batch loading speed.

    Args:
        roi_enabled: Enable ROI extraction.
        num_batches: Number of batches to process.

    Returns:
        Dict with timing statistics.
    """
    # Create dataloaders
    loaders, info = create_openset_data_loaders(
        dataset_name="dorsal",
        img_size=224,
        known_ratio=0.7,
        val_ratio=0.15,
        seed=42,
        P=8,
        K=4,
        num_workers=0,
        batch_size=32,
    )

    train_loader = loaders["train"]

    # Warm-up batch
    next(iter(train_loader))

    # Benchmark
    times = []
    start = time.time()

    for i, batch in enumerate(train_loader):
        if i >= num_batches:
            break
        batch_start = time.time()
        images, labels, metadata = batch
        batch_time = time.time() - batch_start
        times.append(batch_time)

    total_time = time.time() - start

    return {
        "roi_enabled": roi_enabled,
        "num_batches": num_batches,
        "total_time_sec": total_time,
        "avg_batch_time_ms": (total_time / num_batches) * 1000,
        "batches_per_sec": num_batches / total_time,
    }


def main():
    """Run benchmark comparison."""
    print("\n" + "=" * 70)
    print("ROI EXTRACTION - PERFORMANCE BENCHMARK")
    print("=" * 70)

    print("\n⏳ Benchmarking ROI DISABLED...")
    result_off = benchmark_batch_loading(roi_enabled=False, num_batches=20)

    print("\n⏳ Benchmarking ROI ENABLED...")
    result_on = benchmark_batch_loading(roi_enabled=True, num_batches=20)

    # Print results
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    print(f"\n{'Metric':<30} {'ROI OFF':<20} {'ROI ON':<20}")
    print("-" * 70)

    metrics = [
        ("Total time (sec)", result_off["total_time_sec"], result_on["total_time_sec"]),
        (
            "Avg batch time (ms)",
            result_off["avg_batch_time_ms"],
            result_on["avg_batch_time_ms"],
        ),
        (
            "Batches per second",
            result_off["batches_per_sec"],
            result_on["batches_per_sec"],
        ),
    ]

    for metric, val_off, val_on in metrics:
        overhead = ((val_on - val_off) / val_off) * 100 if val_off != 0 else 0
        print(f"{metric:<30} {val_off:<20.3f} {val_on:<20.3f} ({overhead:+.1f}%)")

    # Summary
    print("\n📊 Summary:")
    overhead_pct = (
        (result_on["avg_batch_time_ms"] - result_off["avg_batch_time_ms"])
        / result_off["avg_batch_time_ms"]
    ) * 100
    print(f"  ROI extraction overhead: {overhead_pct:+.1f}%")

    if overhead_pct < 5:
        print("  ✅ Overhead is minimal (<5%)")
    elif overhead_pct < 20:
        print("  ⚠️  Overhead is moderate (5-20%)")
    else:
        print("  ❌ Overhead is significant (>20%)")


if __name__ == "__main__":
    main()
