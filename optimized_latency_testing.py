import time
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import pandas as pd
import argparse
from typing import List, Dict, Any
import json
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

class OptimizedE5LatencyTest:
    def __init__(self, model_path: str, device: str = None):
        """
        Initialize the E5 model for latency testing using SentenceTransformers
        
        Args:
            model_path: Path to the local E5 model directory
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.model_path = model_path
        
        # Set device
        if device:
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        print(f"Using device: {self.device}")
        
        # Load model using SentenceTransformers
        self.load_model()
    
    def load_model(self):
        """Load the E5 model using SentenceTransformers from local path"""
        print(f"Loading model from {self.model_path}")
        
        # Load model with SentenceTransformers
        self.model = SentenceTransformer(self.model_path, device=self.device)
        
        # Configure model for optimized performance
        self.model.max_seq_length = 256  # Reduce sequence length for faster processing
        
        # Perform a warmup inference to initialize everything
        self._warmup()
    
    def _warmup(self, num_warmup: int = 5):
        """
        Perform warmup inference passes
        
        Args:
            num_warmup: Number of warmup passes
        """
        print(f"Performing {num_warmup} warmup passes...")
        sample_text = "What is hypertension and how is it treated?"
        for _ in range(num_warmup):
            self.model.encode(sample_text)
    
    def encode(self, texts: List[str], batch_size: int = 1) -> np.ndarray:
        """
        Encode a list of texts using the E5 model
        
        Args:
            texts: List of input texts to encode
            batch_size: Batch size for encoding
            
        Returns:
            Numpy array of embeddings
        """
        return self.model.encode(texts, batch_size=batch_size)
    
    def measure_latency(self, queries: List[str], batch_size: int = 1, num_runs: int = 100) -> Dict[str, Any]:
        """
        Measure latency for a list of queries
        
        Args:
            queries: List of queries to encode
            batch_size: Batch size to use
            num_runs: Number of test runs
            
        Returns:
            Dictionary with latency statistics
        """
        latencies = []
        
        for _ in tqdm(range(num_runs), desc=f"Testing batch size {batch_size}"):
            # Randomly sample a batch of queries
            indices = np.random.choice(len(queries), batch_size)
            batch_queries = [queries[i] for i in indices]
            
            # Measure encoding time
            start_time = time.time()
            if batch_size == 1:
                self.encode([batch_queries[0]])
            else:
                self.encode(batch_queries, batch_size=batch_size)
            
            latency = (time.time() - start_time) * 1000  # Convert to ms
            latencies.append(latency)
        
        # Calculate statistics
        latencies = np.array(latencies)
        stats = {
            "mean_latency_ms": float(np.mean(latencies)),
            "median_latency_ms": float(np.median(latencies)),
            "std_latency_ms": float(np.std(latencies)),
            "min_latency_ms": float(np.min(latencies)),
            "max_latency_ms": float(np.max(latencies)),
            "p50_latency_ms": float(np.percentile(latencies, 50)),
            "p90_latency_ms": float(np.percentile(latencies, 90)),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
            "p99_latency_ms": float(np.percentile(latencies, 99)),
            "batch_size": batch_size,
            "num_runs": num_runs,
            "device": self.device,
            "raw_latencies": latencies.tolist()  # Store raw data for histograms
        }
        
        return stats

def load_medical_queries(file_path: str = None) -> List[str]:
    """
    Load medical queries for testing
    
    Args:
        file_path: Path to the file containing medical queries
        
    Returns:
        List of medical queries
    """
    # If file path is provided, load from file
    if file_path and os.path.exists(file_path):
        with open(file_path, 'r') as f:
            if file_path.endswith('.json'):
                return json.load(f)
            else:
                return [line.strip() for line in f if line.strip()]
    
    # Otherwise use default medical queries
    return [
        "What are the symptoms of diabetes?",
        "How is rheumatoid arthritis diagnosed?",
        "What are the side effects of beta blockers?",
        "What is the treatment for pneumonia?",
        "How does chemotherapy work?",
        "What are the early signs of Alzheimer's disease?",
        "How is blood pressure measured?",
        "What causes migraines?",
        "What are the different types of vaccines?",
        "How is asthma managed in children?",
        "What are the symptoms of COVID-19?",
        "What is the recommended treatment for hyperthyroidism?",
        "How does insulin resistance develop?",
        "What are the stages of chronic kidney disease?",
        "How is sleep apnea diagnosed?",
        "What are the side effects of statins?",
        "What are the differences between MRI and CT scans?",
        "How do antibiotics work?",
        "What is the mechanism of action for ACE inhibitors?",
        "How is multiple sclerosis diagnosed and treated?",
        "What are the complications of untreated hypertension?",
        "How does the immune system respond to viral infections?",
        "What is the pathophysiology of type 1 diabetes?",
        "How do corticosteroids reduce inflammation?",
        "What are the different classes of antidepressants?",
        "How does hemodialysis work?",
        "What is the normal range for blood glucose levels?",
        "How is heart failure classified?",
        "What are the main types of lung cancer?",
        "How do proton pump inhibitors work to reduce stomach acid?",
        "What are the diagnostic criteria for ADHD?",
        "How do blood thinners prevent clotting?",
        "What is the difference between Type 1 and Type 2 diabetes?",
        "How is bacterial meningitis treated?",
        "What are the stages of chronic obstructive pulmonary disease?",
        "How does radiation therapy target cancer cells?",
        "What is the normal range for thyroid hormone levels?",
        "How are seizures classified?",
        "What is the pathophysiology of heart failure?",
        "How are autoimmune disorders diagnosed?"
    ]

def plot_latency_histogram(results, output_dir):
    """
    Create histograms of latency distributions
    
    Args:
        results: List of result dictionaries
        output_dir: Directory to save plots
    """
    for result in results:
        batch_size = result['batch_size']
        latencies = result['raw_latencies']
        
        plt.figure(figsize=(10, 6))
        plt.hist(latencies, bins=30, alpha=0.7)
        plt.axvline(result['p50_latency_ms'], color='r', linestyle='--', label=f'P50: {result["p50_latency_ms"]:.2f} ms')
        plt.axvline(result['p90_latency_ms'], color='g', linestyle='--', label=f'P90: {result["p90_latency_ms"]:.2f} ms')
        plt.axvline(result['p95_latency_ms'], color='b', linestyle='--', label=f'P95: {result["p95_latency_ms"]:.2f} ms')
        plt.axvline(result['p99_latency_ms'], color='m', linestyle='--', label=f'P99: {result["p99_latency_ms"]:.2f} ms')
        
        plt.title(f'Latency Distribution for Batch Size {batch_size}')
        plt.xlabel('Latency (ms)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        plot_file = os.path.join(output_dir, f'latency_histogram_batch_{batch_size}.png')
        plt.savefig(plot_file)
        plt.close()
        print(f"Histogram saved to {plot_file}")

def plot_comparison_chart(results, output_dir):
    """
    Create a bar chart comparing different batch sizes
    
    Args:
        results: List of result dictionaries
        output_dir: Directory to save plots
    """
    batch_sizes = [r['batch_size'] for r in results]
    mean_latencies = [r['mean_latency_ms'] for r in results]
    p90_latencies = [r['p90_latency_ms'] for r in results]
    p95_latencies = [r['p95_latency_ms'] for r in results]
    p99_latencies = [r['p99_latency_ms'] for r in results]
    
    x = np.arange(len(batch_sizes))
    width = 0.2
    
    fig, ax = plt.figure(figsize=(12, 7)), plt.gca()
    ax.bar(x - width*1.5, mean_latencies, width, label='Mean')
    ax.bar(x - width/2, p90_latencies, width, label='P90')
    ax.bar(x + width/2, p95_latencies, width, label='P95')
    ax.bar(x + width*1.5, p99_latencies, width, label='P99')
    
    ax.set_title('Latency Comparison Across Batch Sizes')
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Latency (ms)')
    ax.set_xticks(x)
    ax.set_xticklabels(batch_sizes)
    ax.legend()
    
    # Add values on top of bars
    for i, v in enumerate(mean_latencies):
        ax.text(i - width*1.5, v + 1, f'{v:.1f}', ha='center', fontsize=9)
    for i, v in enumerate(p90_latencies):
        ax.text(i - width/2, v + 1, f'{v:.1f}', ha='center', fontsize=9)
    for i, v in enumerate(p95_latencies):
        ax.text(i + width/2, v + 1, f'{v:.1f}', ha='center', fontsize=9)
    for i, v in enumerate(p99_latencies):
        ax.text(i + width*1.5, v + 1, f'{v:.1f}', ha='center', fontsize=9)
    
    plt.grid(True, alpha=0.3, axis='y')
    
    # Save the plot
    plot_file = os.path.join(output_dir, 'latency_comparison.png')
    plt.savefig(plot_file)
    plt.close()
    print(f"Comparison chart saved to {plot_file}")

def run_optimized_latency_tests(model_path: str, output_dir: str, batch_sizes: List[int] = [1, 4, 8, 16], 
                               num_runs: int = 100, queries_file: str = None):
    """
    Run latency tests and save results using the optimized SentenceTransformer implementation
    
    Args:
        model_path: Path to the E5 model
        output_dir: Directory to save test results
        batch_sizes: List of batch sizes to test
        num_runs: Number of test runs per batch size
        queries_file: Path to file containing test queries
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load medical queries
    queries = load_medical_queries(queries_file)
    print(f"Loaded {len(queries)} queries for testing")
    
    # Initialize model
    e5_test = OptimizedE5LatencyTest(model_path)
    
    results = []
    
    # Run tests for different batch sizes
    for batch_size in batch_sizes:
        print(f"\nRunning latency test with batch size {batch_size}...")
        stats = e5_test.measure_latency(queries, batch_size=batch_size, num_runs=num_runs)
        
        # Print summary
        print(f"Batch size: {batch_size}")
        print(f"Mean latency: {stats['mean_latency_ms']:.2f} ms")
        print(f"Median latency: {stats['median_latency_ms']:.2f} ms")
        print(f"P90 latency: {stats['p90_latency_ms']:.2f} ms")
        print(f"P95 latency: {stats['p95_latency_ms']:.2f} ms")
        print(f"P99 latency: {stats['p99_latency_ms']:.2f} ms")
        
        # Add to results
        results.append(stats)
    
    # Create visualizations
    plot_latency_histogram(results, output_dir)
    plot_comparison_chart(results, output_dir)
    
    # Save results
    # Remove raw latencies from the DataFrame to keep it clean
    results_for_df = [{k: v for k, v in r.items() if k != 'raw_latencies'} for r in results]
    results_df = pd.DataFrame(results_for_df)
    results_file = os.path.join(output_dir, "optimized_latency_results.csv")
    results_df.to_csv(results_file, index=False)
    print(f"\nResults saved to {results_file}")
    
    # Save detailed results as JSON
    detailed_file = os.path.join(output_dir, "optimized_latency_results.json")
    with open(detailed_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Detailed results saved to {detailed_file}")

def run_comparison_test(baseline_model_path: str, optimized_model_path: str, output_dir: str):
    """
    Run a comparison test between baseline and optimized implementations
    
    Args:
        baseline_model_path: Path to the baseline E5 model
        optimized_model_path: Path to the optimized E5 model (can be the same path)
        output_dir: Directory to save test results
    """
    from latency_testing import E5LatencyTest, load_medical_queries
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load medical queries
    queries = load_medical_queries()
    print(f"Loaded {len(queries)} queries for testing")
    
    # Batch sizes to test
    batch_sizes = [1, 4, 8]
    num_runs = 20  # Smaller number for quick comparison
    
    # Test results
    baseline_results = []
    optimized_results = []
    
    # Run baseline tests
    print("\n=== BASELINE IMPLEMENTATION ===")
    baseline_model = E5LatencyTest(baseline_model_path)
    for batch_size in batch_sizes:
        print(f"\nRunning baseline test with batch size {batch_size}...")
        stats = baseline_model.measure_latency(queries, batch_size=batch_size, num_runs=num_runs)
        baseline_results.append(stats)
    
    # Run optimized tests
    print("\n=== OPTIMIZED IMPLEMENTATION ===")
    optimized_model = OptimizedE5LatencyTest(optimized_model_path)
    for batch_size in batch_sizes:
        print(f"\nRunning optimized test with batch size {batch_size}...")
        stats = optimized_model.measure_latency(queries, batch_size=batch_size, num_runs=num_runs)
        optimized_results.append(stats)
    
    # Create comparison results
    comparison = []
    for i, batch_size in enumerate(batch_sizes):
        baseline = baseline_results[i]
        optimized = optimized_results[i]
        
        improvement = {
            "batch_size": batch_size,
            "baseline_mean_ms": baseline["mean_latency_ms"],
            "optimized_mean_ms": optimized["mean_latency_ms"],
            "mean_speedup": baseline["mean_latency_ms"] / optimized["mean_latency_ms"],
            "baseline_p95_ms": baseline["p95_latency_ms"],
            "optimized_p95_ms": optimized["p95_latency_ms"],
            "p95_speedup": baseline["p95_latency_ms"] / optimized["p95_latency_ms"],
        }
        comparison.append(improvement)
    
    # Save comparison results
    comparison_df = pd.DataFrame(comparison)
    comparison_file = os.path.join(output_dir, "implementation_comparison.csv")
    comparison_df.to_csv(comparison_file, index=False)
    print(f"\nComparison results saved to {comparison_file}")
    
    # Create comparison chart
    plt.figure(figsize=(10, 6))
    x = np.arange(len(batch_sizes))
    width = 0.35
    
    plt.bar(x - width/2, [r["mean_speedup"] for r in comparison], width, label='Mean Speedup')
    plt.bar(x + width/2, [r["p95_speedup"] for r in comparison], width, label='P95 Speedup')
    
    plt.title('Performance Improvement with Sentence Transformers')
    plt.xlabel('Batch Size')
    plt.ylabel('Speedup Factor (higher is better)')
    plt.xticks(x, batch_sizes)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add values on top of bars
    for i, v in enumerate([r["mean_speedup"] for r in comparison]):
        plt.text(i - width/2, v + 0.1, f'{v:.2f}x', ha='center')
    for i, v in enumerate([r["p95_speedup"] for r in comparison]):
        plt.text(i + width/2, v + 0.1, f'{v:.2f}x', ha='center')
    
    # Save comparison chart
    comparison_chart = os.path.join(output_dir, "speedup_comparison.png")
    plt.savefig(comparison_chart)
    plt.close()
    print(f"Comparison chart saved to {comparison_chart}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure latency of E5 model using Sentence Transformers")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the local E5 model directory")
    parser.add_argument("--output_dir", type=str, default="./optimized_results",
                        help="Directory to save test results")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 4, 8, 16],
                        help="Batch sizes to test")
    parser.add_argument("--num_runs", type=int, default=100,
                        help="Number of test runs per batch size")
    parser.add_argument("--queries_file", type=str, default=None,
                        help="Path to file with test queries (JSON or text file, one query per line)")
    parser.add_argument("--run_comparison", action="store_true",
                        help="Run comparison between baseline and optimized implementation")
    parser.add_argument("--baseline_model_path", type=str, 
                        help="Path to the baseline model for comparison (if different)")
    
    args = parser.parse_args()
    
    # Run the optimized tests
    run_optimized_latency_tests(
        model_path=args.model_path,
        output_dir=args.output_dir,
        batch_sizes=args.batch_sizes,
        num_runs=args.num_runs,
        queries_file=args.queries_file
    )
    
    # Run comparison if requested
    if args.run_comparison:
        baseline_path = args.baseline_model_path if args.baseline_model_path else args.model_path
        comparison_dir = os.path.join(args.output_dir, "comparison")
        run_comparison_test(baseline_path, args.model_path, comparison_dir)