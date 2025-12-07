"""Model benchmarking utilities for inference performance and memory footprint."""
import time
import torch
import numpy as np
from pathlib import Path
import json


def count_parameters(model):
    """Count total and trainable parameters.

    Args:
        model: PyTorch model.

    Returns:
        dict: Parameter counts.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'total_params_m': total_params / 1e6,  # In millions
        'trainable_params_m': trainable_params / 1e6
    }


def get_model_size_mb(model_path):
    """Get model file size in MB.

    Args:
        model_path: Path to model file.

    Returns:
        float: Model size in MB.
    """
    return Path(model_path).stat().st_size / (1024 * 1024)


def benchmark_inference_speed(model, input_shape, device, num_warmup=10, num_iterations=100):
    """Benchmark inference speed.

    Args:
        model: PyTorch model.
        input_shape: Input tensor shape (batch_size, channels, height, width).
        device: Device to run on.
        num_warmup: Number of warmup iterations.
        num_iterations: Number of benchmark iterations.

    Returns:
        dict: Inference metrics.
    """
    model.eval()
    model = model.to(device)

    # Create dummy input
    dummy_input = torch.randn(input_shape, device=device)

    # Warmup
    print(f"  Warming up ({num_warmup} iterations)...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)

    # Synchronize for accurate timing
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()

    # Benchmark
    print(f"  Benchmarking ({num_iterations} iterations)...")
    latencies = []

    with torch.no_grad():
        for _ in range(num_iterations):
            start_time = time.perf_counter()
            _ = model(dummy_input)

            # Synchronize
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elif device.type == 'mps':
                torch.mps.synchronize()

            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)  # Convert to ms

    latencies = np.array(latencies)
    batch_size = input_shape[0]

    return {
        'batch_size': batch_size,
        'mean_latency_ms': float(np.mean(latencies)),
        'std_latency_ms': float(np.std(latencies)),
        'median_latency_ms': float(np.median(latencies)),
        'p95_latency_ms': float(np.percentile(latencies, 95)),
        'p99_latency_ms': float(np.percentile(latencies, 99)),
        'min_latency_ms': float(np.min(latencies)),
        'max_latency_ms': float(np.max(latencies)),
        'latency_per_image_ms': float(np.mean(latencies) / batch_size),
        'throughput_img_per_sec': float(batch_size * 1000 / np.mean(latencies)),
        'device': str(device)
    }


def measure_memory_footprint(model, input_shape, device):
    """Measure memory footprint during inference.

    Args:
        model: PyTorch model.
        input_shape: Input tensor shape.
        device: Device to run on.

    Returns:
        dict: Memory metrics.
    """
    model.eval()
    model = model.to(device)

    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Get baseline memory
        baseline_memory = torch.cuda.memory_allocated() / (1024 ** 2)  # MB

        # Run inference
        dummy_input = torch.randn(input_shape, device=device)
        with torch.no_grad():
            _ = model(dummy_input)

        # Get peak memory
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        current_memory = torch.cuda.memory_allocated() / (1024 ** 2)  # MB

        return {
            'baseline_memory_mb': float(baseline_memory),
            'peak_memory_mb': float(peak_memory),
            'current_memory_mb': float(current_memory),
            'inference_memory_mb': float(peak_memory - baseline_memory),
            'device': 'cuda'
        }

    elif device.type == 'mps':
        # MPS doesn't have detailed memory tracking like CUDA
        # Use approximate methods
        return {
            'baseline_memory_mb': None,
            'peak_memory_mb': None,
            'current_memory_mb': None,
            'inference_memory_mb': None,
            'device': 'mps',
            'note': 'MPS memory tracking not available'
        }

    else:
        # CPU - estimate from model parameters
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
        return {
            'parameter_memory_mb': float(param_memory),
            'device': 'cpu',
            'note': 'CPU memory estimate based on parameters only'
        }


def compute_model_flops(model, input_shape):
    """Compute theoretical FLOPs (requires thop or fvcore).

    Args:
        model: PyTorch model.
        input_shape: Input tensor shape.

    Returns:
        dict: FLOPs metrics or None if libraries not available.
    """
    try:
        from thop import profile, clever_format

        dummy_input = torch.randn(input_shape)
        macs, params = profile(model, inputs=(dummy_input,), verbose=False)
        macs_str, params_str = clever_format([macs, params], "%.3f")

        return {
            'flops': float(macs * 2),  # MACs * 2 = FLOPs
            'flops_str': f"{float(macs * 2 / 1e9):.2f}G",  # GFLOPs
            'macs': float(macs),
            'macs_str': macs_str,
            'params_from_flop_counter': params_str
        }
    except ImportError:
        print("  Warning: thop not installed. FLOPs calculation skipped.")
        print("  Install with: pip install thop")
        return None


def benchmark_model(model, model_path, input_shape=(1, 3, 224, 224), device=None):
    """Run complete benchmark on a model.

    Benchmarks on both CPU and GPU (if available) to help with production deployment decisions.
    This allows comparing inference costs between CPU-only and GPU-accelerated deployment targets.

    Args:
        model: PyTorch model.
        model_path: Path to saved model file.
        input_shape: Input tensor shape for benchmarking.
        device: Device to benchmark on (default: benchmark on all available devices).

    Returns:
        dict: Complete benchmark results.
    """
    if device is None:
        # Detect available devices
        devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            devices.append(torch.device('cuda'))
        elif torch.backends.mps.is_available():
            devices.append(torch.device('mps'))
    else:
        devices = [device]

    print(f"\n🔬 Benchmarking model on {len(devices)} device(s): {[str(d) for d in devices]}...")

    results = {
        'benchmarked_devices': [str(d) for d in devices]
    }

    # Model size metrics (device-independent)
    print("  📊 Counting parameters...")
    param_counts = count_parameters(model)
    results['parameters'] = param_counts

    if Path(model_path).exists():
        results['model_size_mb'] = get_model_size_mb(model_path)
        print(f"  💾 Model size: {results['model_size_mb']:.2f} MB")

    # FLOPs computation (device-independent)
    print("  🧮 Computing FLOPs...")
    flops = compute_model_flops(model, input_shape)
    if flops:
        results['flops'] = flops

    # Benchmark on each device
    for bench_device in devices:
        device_name = str(bench_device)
        print(f"\n  🔍 Benchmarking on {device_name.upper()}...")

        device_results = {}

        # Memory footprint
        print(f"    💾 Measuring memory footprint...")
        memory_metrics = measure_memory_footprint(model, input_shape, bench_device)
        device_results['memory'] = memory_metrics

        # Inference speed - batch size 1 (real-world scenario)
        print(f"    ⚡ Benchmarking single image inference...")
        single_metrics = benchmark_inference_speed(
            model, (1, 3, 224, 224), bench_device, num_warmup=10, num_iterations=100
        )
        device_results['inference_single'] = single_metrics

        # Inference speed - batch size 32 (throughput scenario)
        print(f"    ⚡ Benchmarking batch inference...")
        batch_metrics = benchmark_inference_speed(
            model, (32, 3, 224, 224), bench_device, num_warmup=10, num_iterations=50
        )
        device_results['inference_batch32'] = batch_metrics

        # Store results under device key
        results[device_name] = device_results

    return results


def print_benchmark_summary(results):
    """Print formatted benchmark summary.

    Args:
        results: Benchmark results dictionary.
    """
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    # Parameters
    params = results['parameters']
    print(f"\n📊 Model Size:")
    print(f"  Total parameters: {params['total_params']:,} ({params['total_params_m']:.2f}M)")
    print(f"  Trainable parameters: {params['trainable_params']:,} ({params['trainable_params_m']:.2f}M)")

    if 'model_size_mb' in results:
        print(f"  Model file size: {results['model_size_mb']:.2f} MB")

    # FLOPs
    if 'flops' in results and results['flops']:
        flops = results['flops']
        print(f"\n🧮 Computational Complexity:")
        print(f"  FLOPs: {flops['flops_str']}")
        print(f"  MACs: {flops['macs_str']}")

    # Print results for each benchmarked device
    devices = results.get('benchmarked_devices', [])
    for device_name in devices:
        if device_name not in results:
            continue

        device_results = results[device_name]
        print(f"\n{'─' * 80}")
        print(f"DEVICE: {device_name.upper()}")
        print(f"{'─' * 80}")

        # Memory
        memory = device_results['memory']
        print(f"\n💾 Memory Footprint:")
        if memory.get('inference_memory_mb'):
            print(f"  Inference memory: {memory['inference_memory_mb']:.2f} MB")
            print(f"  Peak memory: {memory['peak_memory_mb']:.2f} MB")
        elif 'parameter_memory_mb' in memory:
            print(f"  Parameter memory: {memory['parameter_memory_mb']:.2f} MB")
        elif 'note' in memory:
            print(f"  {memory['note']}")

        # Single image inference
        single = device_results['inference_single']
        print(f"\n⚡ Single Image Inference:")
        print(f"  Mean latency: {single['latency_per_image_ms']:.2f} ms")
        print(f"  Median latency: {single['median_latency_ms']:.2f} ms")
        print(f"  P95 latency: {single['p95_latency_ms']:.2f} ms")
        print(f"  P99 latency: {single['p99_latency_ms']:.2f} ms")

        # Batch inference
        batch = device_results['inference_batch32']
        print(f"\n⚡ Batch Inference (batch_size=32):")
        print(f"  Mean latency per batch: {batch['mean_latency_ms']:.2f} ms")
        print(f"  Latency per image: {batch['latency_per_image_ms']:.2f} ms")
        print(f"  Throughput: {batch['throughput_img_per_sec']:.1f} images/sec")

    print("\n" + "=" * 80 + "\n")


def save_benchmark_results(results, output_path):
    """Save benchmark results to JSON.

    Args:
        results: Benchmark results dictionary.
        output_path: Path to save JSON file.
    """
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"💾 Benchmark results saved to {output_path}")
