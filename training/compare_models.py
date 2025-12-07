"""Compare benchmark results from multiple trained models."""
import json
import sys
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import argparse


def load_benchmark_results(results_dir: Path) -> Dict[str, Any]:
    """Load benchmark results from a training run directory.

    Args:
        results_dir: Path to training results directory.

    Returns:
        dict: Combined training and benchmark results.
    """
    benchmark_path = results_dir / "benchmark_results.json"
    history_path = results_dir / "training_history.txt"

    if not benchmark_path.exists():
        return None

    with open(benchmark_path, 'r') as f:
        benchmark = json.load(f)

    # Extract best validation accuracy from training history
    best_val_acc = 0.0
    if history_path.exists():
        with open(history_path, 'r') as f:
            for line in f:
                if "Best validation accuracy:" in line:
                    best_val_acc = float(line.split(":")[-1].strip().rstrip('%'))
                    break

    return {
        'directory': results_dir.name,
        'timestamp': results_dir.name.split('_')[0],
        'model_name': '_'.join(results_dir.name.split('_')[1:]),
        'best_val_accuracy': best_val_acc,
        'benchmark': benchmark
    }


def find_latest_results(results_root: Path, model_names: List[str] = None) -> Dict[str, Dict]:
    """Find latest training results for each model.

    Args:
        results_root: Root directory containing training results.
        model_names: List of model names to filter (optional).

    Returns:
        dict: Mapping of model_name -> results data.
    """
    if not results_root.exists():
        print(f"❌ Results directory not found: {results_root}")
        return {}

    # Group directories by model name
    model_results = {}

    for result_dir in sorted(results_root.iterdir(), reverse=True):
        if not result_dir.is_dir():
            continue

        # Parse directory name: YYYYMMDDHHMMSS_modelname
        parts = result_dir.name.split('_', 1)
        if len(parts) != 2:
            continue

        timestamp, model_name = parts

        # Skip if filtering by model names
        if model_names and model_name not in model_names:
            continue

        # Only keep latest result per model
        if model_name in model_results:
            continue

        data = load_benchmark_results(result_dir)
        if data:
            model_results[model_name] = data

    return model_results


def format_memory(memory_mb: float) -> str:
    """Format memory size."""
    if memory_mb is None:
        return "N/A"
    return f"{memory_mb:.1f} MB"


def format_latency(latency_ms: float) -> str:
    """Format latency."""
    return f"{latency_ms:.2f} ms"


def format_throughput(throughput: float) -> str:
    """Format throughput."""
    return f"{throughput:.1f} img/s"


def print_comparison_table(results: Dict[str, Dict]):
    """Print formatted comparison table.

    Args:
        results: Dictionary of model results.
    """
    print("\n" + "=" * 140)
    print("MODEL COMPARISON REPORT")
    print("=" * 140)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Models compared: {len(results)}")
    print("=" * 140)

    # Sort by validation accuracy (descending)
    sorted_models = sorted(results.items(),
                          key=lambda x: x[1]['best_val_accuracy'],
                          reverse=True)

    # Header
    print(f"\n{'Model':<25} {'Val Acc':<10} {'Params':<12} {'Size':<10} {'FLOPs':<10}")
    print("-" * 140)

    # Model overview
    for model_name, data in sorted_models:
        benchmark = data['benchmark']
        params = benchmark['parameters']

        acc_str = f"{data['best_val_accuracy']:.2f}%"
        params_str = f"{params['total_params_m']:.2f}M"
        size_str = f"{benchmark.get('model_size_mb', 0):.1f} MB"

        flops_str = "N/A"
        if 'flops' in benchmark and benchmark['flops']:
            flops_str = benchmark['flops']['flops_str']

        print(f"{model_name:<25} {acc_str:<10} {params_str:<12} {size_str:<10} {flops_str:<10}")

    print("\n" + "=" * 140)

    # Detailed performance comparison
    devices = []
    # Determine which devices were benchmarked
    if sorted_models:
        first_benchmark = sorted_models[0][1]['benchmark']
        devices = first_benchmark.get('benchmarked_devices', [])

    for device in devices:
        print(f"\n{'─' * 140}")
        print(f"INFERENCE PERFORMANCE: {device.upper()}")
        print(f"{'─' * 140}")

        # Headers
        print(f"\n{'Model':<25} {'Single Img':<15} {'P95 Latency':<15} {'Batch-32':<15} {'Throughput':<15} {'Memory':<15}")
        print("-" * 140)

        for model_name, data in sorted_models:
            benchmark = data['benchmark']

            if device not in benchmark:
                print(f"{model_name:<25} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<15}")
                continue

            device_data = benchmark[device]
            single = device_data['inference_single']
            batch = device_data['inference_batch32']
            memory = device_data['memory']

            single_lat = format_latency(single['latency_per_image_ms'])
            p95_lat = format_latency(single['p95_latency_ms'])
            batch_lat = format_latency(batch['latency_per_image_ms'])
            throughput = format_throughput(batch['throughput_img_per_sec'])

            mem_str = "N/A"
            if memory.get('inference_memory_mb'):
                mem_str = format_memory(memory['inference_memory_mb'])
            elif memory.get('parameter_memory_mb'):
                mem_str = format_memory(memory['parameter_memory_mb'])

            print(f"{model_name:<25} {single_lat:<15} {p95_lat:<15} {batch_lat:<15} {throughput:<15} {mem_str:<15}")

    print("\n" + "=" * 140)

    # Recommendations
    print("\n📊 RECOMMENDATIONS")
    print("=" * 140)

    best_accuracy = sorted_models[0]
    print(f"🏆 Best Accuracy: {best_accuracy[0]} ({best_accuracy[1]['best_val_accuracy']:.2f}%)")

    # Find smallest model
    smallest = min(sorted_models, key=lambda x: x[1]['benchmark']['parameters']['total_params_m'])
    print(f"📦 Smallest Model: {smallest[0]} ({smallest[1]['benchmark']['parameters']['total_params_m']:.2f}M params)")

    # Find fastest on CPU (if available)
    if 'cpu' in devices:
        fastest_cpu = min(sorted_models,
                         key=lambda x: x[1]['benchmark'].get('cpu', {}).get('inference_single', {}).get('latency_per_image_ms', float('inf')))
        cpu_latency = fastest_cpu[1]['benchmark']['cpu']['inference_single']['latency_per_image_ms']
        print(f"⚡ Fastest CPU Inference: {fastest_cpu[0]} ({cpu_latency:.2f} ms/image)")

    # Find fastest on GPU (if available)
    gpu_device = next((d for d in devices if d in ['cuda', 'mps']), None)
    if gpu_device:
        fastest_gpu = min(sorted_models,
                         key=lambda x: x[1]['benchmark'].get(gpu_device, {}).get('inference_single', {}).get('latency_per_image_ms', float('inf')))
        gpu_latency = fastest_gpu[1]['benchmark'][gpu_device]['inference_single']['latency_per_image_ms']
        print(f"🚀 Fastest GPU Inference: {fastest_gpu[0]} ({gpu_latency:.2f} ms/image)")

    # Best balance
    print(f"\n💡 Balanced Choice:")
    for model_name, data in sorted_models[:3]:
        acc = data['best_val_accuracy']
        params = data['benchmark']['parameters']['total_params_m']

        if 'cpu' in devices:
            cpu_lat = data['benchmark']['cpu']['inference_single']['latency_per_image_ms']
            print(f"   {model_name}: {acc:.2f}% accuracy, {params:.1f}M params, {cpu_lat:.1f}ms CPU latency")
        else:
            print(f"   {model_name}: {acc:.2f}% accuracy, {params:.1f}M params")

    print("\n" + "=" * 140 + "\n")


def save_comparison_json(results: Dict[str, Dict], output_path: Path):
    """Save comparison results to JSON.

    Args:
        results: Dictionary of model results.
        output_path: Path to save JSON file.
    """
    output_path.write_text(json.dumps(results, indent=2))
    print(f"💾 Comparison results saved to {output_path}")


def main():
    """Main comparison function."""
    parser = argparse.ArgumentParser(description="Compare trained model results")
    parser.add_argument('--results-dir', type=str,
                       help='Results root directory (default: from RESULTS_ROOT_DIR env)')
    parser.add_argument('--models', nargs='+',
                       help='Specific model names to compare (default: all)')
    parser.add_argument('--output', type=str, default='model_comparison.json',
                       help='Output JSON file path')

    args = parser.parse_args()

    # Get results directory
    if args.results_dir:
        results_root = Path(args.results_dir).expanduser()
    else:
        import os
        from dotenv import load_dotenv
        load_dotenv()
        results_root_str = os.getenv('RESULTS_ROOT_DIR')
        if not results_root_str:
            print("❌ Error: RESULTS_ROOT_DIR not set in environment")
            sys.exit(1)
        results_root = Path(results_root_str).expanduser()

    print(f"🔍 Searching for results in: {results_root}")

    # Find and load results
    results = find_latest_results(results_root, args.models)

    if not results:
        print("❌ No training results found")
        sys.exit(1)

    print(f"✓ Found results for {len(results)} model(s)")

    # Print comparison table
    print_comparison_table(results)

    # Save JSON
    output_path = Path(args.output)
    save_comparison_json(results, output_path)


if __name__ == '__main__':
    main()
