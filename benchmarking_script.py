#!/usr/bin/env python
"""
Comprehensive E5 Model Benchmark Script
======================================

This script provides a complete benchmark for E5 models, testing:
- Baseline vs optimized implementations
- Various batch sizes
- Different optimization techniques
- End-to-end performance metrics

Author: Machine Learning Engineer
Date: April 14, 2025
"""

import os
import sys
import time
import argparse
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

# Optional imports - will be checked dynamically
try:
    from transformers import AutoModel, AutoTokenizer
    HAVE_TRANSFORMERS = True
except ImportError:
    HAVE_TRANSFORMERS = False

try:
    from sentence_transformers import SentenceTransformer
    HAVE_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAVE_SENTENCE_TRANSFORMERS = False



class BaselineE5Benchmark:
    """Benchmark using standard Hugging Face transformers"""
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        Initialize the baseline E5 benchmark
        
        Args:
            model_path: Path to the local E5 model directory
            device: Device to run the model on ('cuda' or 'cpu')
        """
        if not HAVE_TRANSFORMERS:
            raise ImportError("transformers library not found. Install with: pip install transformers")
        
        self.model_path = model_path
        
        # Set device
        if device:
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        print(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self.load_model()
    
    def load_model(self):
        """Load the E5 model and tokenizer from local path"""
        print(f"Loading model from {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModel.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
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
            with torch.no_grad():
                inputs = self.tokenizer(sample_text, return_tensors="pt").to(self.device)
                self.model(**inputs)
    
    def encode(self, text: str or List[str], batch_size: int = 1) -> np.ndarray:
        """
        Encode a text using the E5 model
        
        Args:
            text: Input text or list of texts to encode
            batch_size: Batch size for processing
            
        Returns:
            Numpy array of embeddings
        """
        # Handle single string vs list of strings
        is_single = isinstance(text, str)
        texts = [text] if is_single else text
        
        # Process in batches
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            with torch.no_grad():
                inputs = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
                # Use the mean pooling of the last hidden state as the embedding
                embeddings = outputs.last_hidden_state.mean(dim=1)
                all_embeddings.append(embeddings.cpu().numpy())
        
        # Concatenate results
        combined_embeddings = np.vstack(all_embeddings)
        
        # Return single result or batch
        return combined_embeddings[0] if is_single else combined_embeddings
    
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
            self.encode(batch_queries, batch_size=batch_size)
            latency = (time.time() - start_time) * 1000  # Convert to ms
            latencies.append(latency)
        
        # Calculate statistics
        latencies = np.array(latencies)
        stats = {
            "implementation": "baseline",
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
            "raw_latencies": latencies.tolist()
        }
        
        return stats


class OptimizedE5Benchmark:
    """Benchmark using optimized SentenceTransformers"""
    
    def __init__(self, model_path: str, device: Optional[str] = None, max_seq_length: int = 256):
        """
        Initialize the optimized E5 benchmark
        
        Args:
            model_path: Path to the local E5 model directory
            device: Device to run the model on ('cuda' or 'cpu')
            max_seq_length: Maximum sequence length
        """
        if not HAVE_SENTENCE_TRANSFORMERS:
            raise ImportError("sentence_transformers library not found. Install with: pip install sentence-transformers")
        
        self.model_path = model_path
        self.max_seq_length = max_seq_length
        
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
        self.model.max_seq_length = self.max_seq_length
        
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
    
    def encode(self, text: str or List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode a text using the E5 model
        
        Args:
            text: Input text or list of texts to encode
            batch_size: Batch size for processing
            
        Returns:
            Numpy array of embeddings
        """
        # SentenceTransformers handles batching internally
        return self.model.encode(text, batch_size=batch_size)
    
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
            self.encode(batch_queries, batch_size=batch_size)
            latency = (time.time() - start_time) * 1000  # Convert to ms
            latencies.append(latency)
        
        # Calculate statistics
        latencies = np.array(latencies)
        stats = {
            "implementation": "optimized",
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
            "max_seq_length": self.max_seq_length,
            "raw_latencies": latencies.tolist()
        }
        
        return stats


class OptimizedConfigurationBenchmark:
    """Benchmark with different optimization configurations"""
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        Initialize the optimized configuration benchmark
        
        Args:
            model_path: Path to the local E5 model directory
            device: Device to run the model on ('cuda' or 'cpu')
        """
        if not HAVE_SENTENCE_TRANSFORMERS:
            raise ImportError("sentence_transformers library not found. Install with: pip install sentence-transformers")
        
        self.model_path = model_path
        
        # Set device
        if device:
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        print(f"Using device: {self.device}")
    
    def apply_optimization(self, technique: str, **kwargs) -> SentenceTransformer:
        """
        Apply specific optimization technique to the model
        
        Args:
            technique: Optimization technique to apply
            **kwargs: Additional parameters for the technique
            
        Returns:
            Optimized SentenceTransformer model
        """
        print(f"Applying optimization: {technique} with params: {kwargs}")
        
        # Load base model
        model = SentenceTransformer(self.model_path, device=self.device)
        
        if technique == "sequence_length":
            # Adjust sequence length
            max_seq_length = kwargs.get("max_seq_length", 128)
            model.max_seq_length = max_seq_length
            
        elif technique == "fp16":
            # Convert to half precision
            if self.device == 'cuda':  # Only on GPU
                model.half()
                
        elif technique == "normalized_embeddings":
            # Just configure normalization (will be applied at encoding time)
            pass
            
        else:
            print(f"Unknown optimization technique: {technique}")
        
        # Warmup
        sample_text = "What is hypertension and how is it treated?"
        for _ in range(3):
            model.encode(sample_text)
            
        return model
    
    def benchmark_technique(self, technique: str, queries: List[str], 
                           batch_sizes: List[int], num_runs: int, **kwargs) -> List[Dict[str, Any]]:
        """
        Benchmark a specific optimization technique
        
        Args:
            technique: Optimization technique to apply
            queries: List of queries to encode
            batch_sizes: List of batch sizes to test
            num_runs: Number of test runs per batch size
            **kwargs: Additional parameters for the technique
            
        Returns:
            List of dictionaries with latency statistics
        """
        # Apply optimization
        model = self.apply_optimization(technique, **kwargs)
        
        results = []
        
        # Test each batch size
        for batch_size in batch_sizes:
            print(f"\nTesting {technique} with batch size {batch_size}...")
            
            latencies = []
            
            for _ in tqdm(range(num_runs), desc=f"Testing batch size {batch_size}"):
                # Randomly sample a batch of queries
                indices = np.random.choice(len(queries), batch_size)
                batch_queries = [queries[i] for i in indices]
                
                # Handle normalized_embeddings technique
                normalize = technique == "normalized_embeddings"
                
                # Measure encoding time
                start_time = time.time()
                model.encode(batch_queries, batch_size=batch_size, normalize_embeddings=normalize)
                latency = (time.time() - start_time) * 1000  # Convert to ms
                latencies.append(latency)
            
            # Calculate statistics
            latencies = np.array(latencies)
            stats = {
                "implementation": "optimized",
                "optimization": technique,
                "optimization_params": str(kwargs),
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
                "raw_latencies": latencies.tolist()
            }
            
            results.append(stats)
            
            # Print summary
            print(f"  Mean latency: {stats['mean_latency_ms']:.2f} ms")
            print(f"  P95 latency: {stats['p95_latency_ms']:.2f} ms")
            print(f"  P99 latency: {stats['p99_latency_ms']:.2f} ms")
        
        return results


def load_medical_queries(file_path: Optional[str] = None) -> List[str]:
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


def plot_latency_histogram(results: List[Dict[str, Any]], output_dir: str):
    """
    Create histograms of latency distributions
    
    Args:
        results: List of result dictionaries
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for result in results:
        impl = result['implementation']
        batch_size = result['batch_size']
        latencies = result['raw_latencies']
        optimization = result.get('optimization', 'none')
        
        plt.figure(figsize=(10, 6))
        plt.hist(latencies, bins=30, alpha=0.7)
        plt.axvline(result['p50_latency_ms'], color='r', linestyle='--', label=f'P50: {result["p50_latency_ms"]:.2f} ms')
        plt.axvline(result['p90_latency_ms'], color='g', linestyle='--', label=f'P90: {result["p90_latency_ms"]:.2f} ms')
        plt.axvline(result['p95_latency_ms'], color='b', linestyle='--', label=f'P95: {result["p95_latency_ms"]:.2f} ms')
        plt.axvline(result['p99_latency_ms'], color='m', linestyle='--', label=f'P99: {result["p99_latency_ms"]:.2f} ms')
        
        title = f'{impl.capitalize()} Implementation - Batch Size {batch_size}'
        if optimization != 'none':
            title += f' - {optimization}'
            
        plt.title(title)
        plt.xlabel('Latency (ms)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        if optimization != 'none':
            filename = f'latency_histogram_{impl}_{optimization}_batch_{batch_size}.png'
        else:
            filename = f'latency_histogram_{impl}_batch_{batch_size}.png'
            
        plot_file = os.path.join(output_dir, filename)
        plt.savefig(plot_file)
        plt.close()
        print(f"Histogram saved to {plot_file}")


def plot_batch_size_comparison(results: List[Dict[str, Any]], output_dir: str):
    """
    Create comparison charts for different batch sizes
    
    Args:
        results: List of result dictionaries
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Group results by implementation
    implementations = {}
    for result in results:
        impl = result['implementation']
        if impl not in implementations:
            implementations[impl] = []
        implementations[impl].append(result)
    
    # Plot mean latency vs batch size
    plt.figure(figsize=(12, 7))
    
    for impl, impl_results in implementations.items():
        # Sort by batch size
        impl_results.sort(key=lambda x: x['batch_size'])
        
        batch_sizes = [r['batch_size'] for r in impl_results]
        mean_latencies = [r['mean_latency_ms'] for r in impl_results]
        
        plt.plot(batch_sizes, mean_latencies, 'o-', label=f'{impl.capitalize()}')
    
    plt.title('Mean Latency vs Batch Size')
    plt.xlabel('Batch Size')
    plt.ylabel('Mean Latency (ms)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Use log scale for x-axis if batch sizes span multiple orders of magnitude
    if max(batch_sizes) / min(batch_sizes) > 10:
        plt.xscale('log')
    
    plot_file = os.path.join(output_dir, 'batch_size_comparison.png')
    plt.savefig(plot_file)
    plt.close()
    print(f"Batch size comparison saved to {plot_file}")
    
    # Plot P95 latency vs batch size
    plt.figure(figsize=(12, 7))
    
    for impl, impl_results in implementations.items():
        # Sort by batch size
        impl_results.sort(key=lambda x: x['batch_size'])
        
        batch_sizes = [r['batch_size'] for r in impl_results]
        p95_latencies = [r['p95_latency_ms'] for r in impl_results]
        
        plt.plot(batch_sizes, p95_latencies, 'o-', label=f'{impl.capitalize()}')
    
    plt.title('P95 Latency vs Batch Size')
    plt.xlabel('Batch Size')
    plt.ylabel('P95 Latency (ms)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Use log scale for x-axis if batch sizes span multiple orders of magnitude
    if max(batch_sizes) / min(batch_sizes) > 10:
        plt.xscale('log')
    
    plot_file = os.path.join(output_dir, 'p95_batch_size_comparison.png')
    plt.savefig(plot_file)
    plt.close()
    print(f"P95 batch size comparison saved to {plot_file}")


def plot_throughput_comparison(results: List[Dict[str, Any]], output_dir: str):
    """
    Create throughput comparison charts
    
    Args:
        results: List of result dictionaries
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate throughput (queries per second) for each result
    for result in results:
        result['throughput_qps'] = result['batch_size'] * 1000 / result['mean_latency_ms']
    
    # Group results by implementation
    implementations = {}
    for result in results:
        impl = result['implementation']
        if impl not in implementations:
            implementations[impl] = []
        implementations[impl].append(result)
    
    # Plot throughput vs batch size
    plt.figure(figsize=(12, 7))
    
    for impl, impl_results in implementations.items():
        # Sort by batch size
        impl_results.sort(key=lambda x: x['batch_size'])
        
        batch_sizes = [r['batch_size'] for r in impl_results]
        throughputs = [r['throughput_qps'] for r in impl_results]
        
        plt.plot(batch_sizes, throughputs, 'o-', label=f'{impl.capitalize()}')
    
    plt.title('Throughput vs Batch Size')
    plt.xlabel('Batch Size')
    plt.ylabel('Throughput (queries/second)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Use log scale for x-axis if batch sizes span multiple orders of magnitude
    if max(batch_sizes) / min(batch_sizes) > 10:
        plt.xscale('log')
    
    plot_file = os.path.join(output_dir, 'throughput_comparison.png')
    plt.savefig(plot_file)
    plt.close()
    print(f"Throughput comparison saved to {plot_file}")


def plot_optimization_comparison(results: List[Dict[str, Any]], output_dir: str):
    """
    Create comparison charts for different optimization techniques
    
    Args:
        results: List of result dictionaries
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter results with optimization field
    optimization_results = [r for r in results if 'optimization' in r]
    
    if not optimization_results:
        return
    
    # Group by batch size
    batch_sizes = {}
    for result in optimization_results:
        bs = result['batch_size']
        if bs not in batch_sizes:
            batch_sizes[bs] = []
        batch_sizes[bs].append(result)
    
    # Plot for each batch size
    for bs, bs_results in batch_sizes.items():
        plt.figure(figsize=(12, 7))
        
        # Sort by optimization technique
        optimizations = sorted(set(r['optimization'] for r in bs_results))
        
        # Extract data for plotting
        mean_latencies = []
        p95_latencies = []
        labels = []
        
        for opt in optimizations:
            # Find result with this optimization
            for result in bs_results:
                if result['optimization'] == opt:
                    mean_latencies.append(result['mean_latency_ms'])
                    p95_latencies.append(result['p95_latency_ms'])
                    labels.append(opt)
                    break
        
        # Plot
        x = np.arange(len(labels))
        width = 0.35
        
        plt.bar(x - width/2, mean_latencies, width, label='Mean Latency')
        plt.bar(x + width/2, p95_latencies, width, label='P95 Latency')
        
        plt.title(f'Optimization Comparison - Batch Size {bs}')
        plt.xlabel('Optimization Technique')
        plt.ylabel('Latency (ms)')
        plt.xticks(x, labels)
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        
        plot_file = os.path.join(output_dir, f'optimization_comparison_batch_{bs}.png')
        plt.savefig(plot_file)
        plt.close()
        print(f"Optimization comparison for batch size {bs} saved to {plot_file}")
    
    # Plot throughput by optimization and batch size
    # Calculate throughput for optimization results
    for result in optimization_results:
        result['throughput_qps'] = result['batch_size'] * 1000 / result['mean_latency_ms']
    
    # Group by optimization
    optimizations = {}
    for result in optimization_results:
        opt = result['optimization']
        if opt not in optimizations:
            optimizations[opt] = []
        optimizations[opt].append(result)
    
    plt.figure(figsize=(12, 7))
    
    for opt, opt_results in optimizations.items():
        # Sort by batch size
        opt_results.sort(key=lambda x: x['batch_size'])
        
        batch_sizes = [r['batch_size'] for r in opt_results]
        throughputs = [r['throughput_qps'] for r in opt_results]
        
        plt.plot(batch_sizes, throughputs, 'o-', label=opt)
    
    plt.title('Throughput by Optimization Technique')
    plt.xlabel('Batch Size')
    plt.ylabel('Throughput (queries/second)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Use log scale for x-axis if batch sizes span multiple orders of magnitude
    if max(batch_sizes) / min(batch_sizes) > 10:
        plt.xscale('log')
    
    plot_file = os.path.join(output_dir, 'optimization_throughput_comparison.png')
    plt.savefig(plot_file)
    plt.close()
    print(f"Optimization throughput comparison saved to {plot_file}")


def generate_report(results: List[Dict[str, Any]], output_dir: str):
    """
    Generate a comprehensive benchmark report
    
    Args:
        results: List of result dictionaries
        output_dir: Directory to save the report
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create Markdown report
    report_path = os.path.join(output_dir, "benchmark_report.md")
    
    with open(report_path, 'w') as f:
        # Header
        f.write("# E5 Model Benchmark Report\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # System information
        f.write("## System Information\n\n")
        f.write(f"- Python version: {sys.version.split()[0]}\n")
        f.write(f"- PyTorch version: {torch.__version__}\n")
        if HAVE_TRANSFORMERS:
            import transformers
            f.write(f"- Transformers version: {transformers.__version__}\n")
        if HAVE_SENTENCE_TRANSFORMERS:
            import sentence_transformers
            f.write(f"- SentenceTransformers version: {sentence_transformers.__version__}\n")
        f.write(f"- Device: {results[0]['device']}\n\n")
        
        # Summary of results
        f.write("## Summary\n\n")
        
        # Group results by implementation
        implementations = {}
        for result in results:
            impl = result['implementation']
            if impl not in implementations:
                implementations[impl] = []
            implementations[impl].append(result)
        
        # Best results by implementation
        f.write("### Best Results by Implementation\n\n")
        f.write("| Implementation | Best Mean Latency (ms) | Best P95 Latency (ms) | Best Throughput (q/s) | Best Batch Size |\n")
        f.write("|----------------|------------------------|------------------------|------------------------|----------------|\n")
        
        for impl, impl_results in implementations.items():
            # Find best mean latency
            best_mean = min(impl_results, key=lambda x: x['mean_latency_ms'])
            best_p95 = min(impl_results, key=lambda x: x['p95_latency_ms'])
            
            # Calculate throughput
            for r in impl_results:
                r['throughput_qps'] = r['batch_size'] * 1000 / r['mean_latency_ms']
            best_throughput = max(impl_results, key=lambda x: x['throughput_qps'])
            
            f.write(f"| {impl.capitalize()} | {best_mean['mean_latency_ms']:.2f} (batch={best_mean['batch_size']}) | ")
            f.write(f"{best_p95['p95_latency_ms']:.2f} (batch={best_p95['batch_size']}) | ")
            f.write(f"{best_throughput['throughput_qps']:.2f} | {best_throughput['batch_size']} |\n")
        
        f.write("\n")
        
        # Visualizations section
        f.write("## Visualizations\n\n")
        f.write("### Latency by Batch Size\n\n")
        f.write("![Batch Size Comparison](./plots/batch_size_comparison.png)\n\n")
        f.write("![P95 Batch Size Comparison](./plots/p95_batch_size_comparison.png)\n\n")
        
        f.write("### Throughput by Batch Size\n\n")
        f.write("![Throughput Comparison](./plots/throughput_comparison.png)\n\n")
        
        # Only include optimization section if we have optimization results
        if any('optimization' in r for r in results):
            f.write("### Optimization Techniques Comparison\n\n")
            f.write("![Optimization Throughput Comparison](./plots/optimization_throughput_comparison.png)\n\n")
            
            # Find unique batch sizes with optimization results
            batch_sizes = set()
            for r in results:
                if 'optimization' in r:
                    batch_sizes.add(r['batch_size'])
            
            for bs in sorted(batch_sizes):
                f.write(f"#### Batch Size {bs}\n\n")
                f.write(f"![Optimization Comparison Batch {bs}](./plots/optimization_comparison_batch_{bs}.png)\n\n")
        
        # Detailed results section
        f.write("## Detailed Results\n\n")
        
        for impl, impl_results in implementations.items():
            f.write(f"### {impl.capitalize()} Implementation\n\n")
            f.write("| Batch Size | Mean Latency (ms) | Median Latency (ms) | P95 Latency (ms) | P99 Latency (ms) | Throughput (q/s) |\n")
            f.write("|------------|-------------------|---------------------|------------------|------------------|------------------|\n")
            
            # Sort by batch size
            sorted_results = sorted(impl_results, key=lambda x: x['batch_size'])
            
            for result in sorted_results:
                optimization = result.get('optimization', None)
                bs = result['batch_size']
                mean = result['mean_latency_ms']
                median = result['median_latency_ms']
                p95 = result['p95_latency_ms']
                p99 = result['p99_latency_ms']
                throughput = result['throughput_qps']
                
                line = f"| {bs} | {mean:.2f} | {median:.2f} | {p95:.2f} | {p99:.2f} | {throughput:.2f} |"
                
                if optimization:
                    line = f"| {bs} ({optimization}) | {mean:.2f} | {median:.2f} | {p95:.2f} | {p99:.2f} | {throughput:.2f} |"
                    
                f.write(line + "\n")
            
            f.write("\n")
        
        # Recommendations
        f.write("## Recommendations\n\n")
        
        # Find overall best configuration
        all_with_throughput = []
        for result in results:
            result_copy = result.copy()
            result_copy['throughput_qps'] = result['batch_size'] * 1000 / result['mean_latency_ms']
            all_with_throughput.append(result_copy)
        
        best_throughput = max(all_with_throughput, key=lambda x: x['throughput_qps'])
        best_latency = min(results, key=lambda x: x['mean_latency_ms'])
        best_p95 = min(results, key=lambda x: x['p95_latency_ms'])
        
        f.write("### For Lowest Latency\n\n")
        f.write(f"- Use the **{best_latency['implementation'].capitalize()}** implementation\n")
        if 'optimization' in best_latency:
            f.write(f"- Apply the **{best_latency['optimization']}** optimization\n")
        f.write(f"- Use batch size **{best_latency['batch_size']}**\n")
        f.write(f"- Expected mean latency: **{best_latency['mean_latency_ms']:.2f} ms**\n")
        f.write(f"- Expected P95 latency: **{best_latency['p95_latency_ms']:.2f} ms**\n\n")
        
        f.write("### For Highest Throughput\n\n")
        f.write(f"- Use the **{best_throughput['implementation'].capitalize()}** implementation\n")
        if 'optimization' in best_throughput:
            f.write(f"- Apply the **{best_throughput['optimization']}** optimization\n")
        f.write(f"- Use batch size **{best_throughput['batch_size']}**\n")
        f.write(f"- Expected throughput: **{best_throughput['throughput_qps']:.2f} queries/second**\n")
        f.write(f"- Expected mean latency: **{best_throughput['mean_latency_ms']:.2f} ms**\n\n")
        
        f.write("### For P95 Latency SLA\n\n")
        f.write(f"- Use the **{best_p95['implementation'].capitalize()}** implementation\n")
        if 'optimization' in best_p95:
            f.write(f"- Apply the **{best_p95['optimization']}** optimization\n")
        f.write(f"- Use batch size **{best_p95['batch_size']}**\n")
        f.write(f"- Expected P95 latency: **{best_p95['p95_latency_ms']:.2f} ms**\n")
        f.write(f"- Expected mean latency: **{best_p95['mean_latency_ms']:.2f} ms**\n\n")
        
        # Conclusion
        f.write("## Conclusion\n\n")
        
        # Compare baseline vs optimized
        baseline_results = implementations.get('baseline', [])
        optimized_results = implementations.get('optimized', [])
        
        if baseline_results and optimized_results:
            # Find comparable batch sizes
            common_batch_sizes = set([r['batch_size'] for r in baseline_results]) & set([r['batch_size'] for r in optimized_results])
            
            if common_batch_sizes:
                improvements = []
                
                for bs in common_batch_sizes:
                    baseline = next(r for r in baseline_results if r['batch_size'] == bs)
                    optimized = next(r for r in optimized_results if r['batch_size'] == bs)
                    
                    mean_improvement = baseline['mean_latency_ms'] / optimized['mean_latency_ms']
                    p95_improvement = baseline['p95_latency_ms'] / optimized['p95_latency_ms']
                    
                    improvements.append({
                        'batch_size': bs,
                        'mean_improvement': mean_improvement,
                        'p95_improvement': p95_improvement
                    })
                
                avg_mean_improvement = sum(i['mean_improvement'] for i in improvements) / len(improvements)
                avg_p95_improvement = sum(i['p95_improvement'] for i in improvements) / len(improvements)
                
                f.write(f"The optimized implementation provides an average of **{avg_mean_improvement:.2f}x** improvement ")
                f.write(f"in mean latency and **{avg_p95_improvement:.2f}x** improvement in P95 latency compared to the baseline implementation.\n\n")
        
        # Final recommendation
        f.write("### Final Recommendation\n\n")
        f.write("Based on the benchmark results, we recommend:\n\n")
        
        if best_throughput['implementation'] == 'optimized':
            f.write("- Use the **SentenceTransformers** implementation for optimal performance\n")
            if 'optimization' in best_throughput:
                f.write(f"- Apply the **{best_throughput['optimization']}** optimization\n")
            if 'max_seq_length' in best_throughput:
                f.write(f"- Set max_seq_length to **{best_throughput['max_seq_length']}**\n")
            f.write(f"- Use batch size **{best_throughput['batch_size']}** for highest throughput\n")
            f.write(f"  - This provides {best_throughput['throughput_qps']:.2f} queries/second with {best_throughput['mean_latency_ms']:.2f} ms mean latency\n")
            
    print(f"Benchmark report generated at {report_path}")
    return report_path


def run_benchmark(model_path: str, output_dir: str, batch_sizes: List[int], num_runs: int, run_baseline: bool = True,
                 run_optimized: bool = True, run_optimizations: bool = True, queries_file: str = None):
    """
    Run comprehensive E5 model benchmark
    
    Args:
        model_path: Path to the E5 model
        output_dir: Directory to save benchmark results
        batch_sizes: List of batch sizes to test
        num_runs: Number of test runs per configuration
        run_baseline: Whether to run baseline implementation
        run_optimized: Whether to run optimized implementation
        run_optimizations: Whether to run optimization techniques
        queries_file: Path to file with test queries
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"e5_benchmark_{timestamp}")
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load queries
    queries = load_medical_queries(queries_file)
    print(f"Loaded {len(queries)} queries for testing")
    
    all_results = []
    
    # Run baseline tests
    if run_baseline and HAVE_TRANSFORMERS:
        print("\n=== Running Baseline Implementation Tests ===\n")
        baseline = BaselineE5Benchmark(model_path)
        
        for batch_size in batch_sizes:
            print(f"\nTesting batch size {batch_size}...")
            result = baseline.measure_latency(queries, batch_size=batch_size, num_runs=num_runs)
            all_results.append(result)
            
            # Print summary
            print(f"  Mean latency: {result['mean_latency_ms']:.2f} ms")
            print(f"  P95 latency: {result['p95_latency_ms']:.2f} ms")
            print(f"  P99 latency: {result['p99_latency_ms']:.2f} ms")
    
    # Run optimized tests
    if run_optimized and HAVE_SENTENCE_TRANSFORMERS:
        print("\n=== Running Optimized Implementation Tests ===\n")
        optimized = OptimizedE5Benchmark(model_path)
        
        for batch_size in batch_sizes:
            print(f"\nTesting batch size {batch_size}...")
            result = optimized.measure_latency(queries, batch_size=batch_size, num_runs=num_runs)
            all_results.append(result)
            
            # Print summary
            print(f"  Mean latency: {result['mean_latency_ms']:.2f} ms")
            print(f"  P95 latency: {result['p95_latency_ms']:.2f} ms")
            print(f"  P99 latency: {result['p99_latency_ms']:.2f} ms")
    
    # Run optimization techniques
    if run_optimizations and HAVE_SENTENCE_TRANSFORMERS:
        print("\n=== Running Optimization Techniques ===\n")
        optim_bench = OptimizedConfigurationBenchmark(model_path)
        
        # Define optimization techniques to test
        optimization_techniques = [
            {
                'name': 'sequence_length',
                'kwargs': {'max_seq_length': 128}
            },
            {
                'name': 'sequence_length',
                'kwargs': {'max_seq_length': 64}
            },
            {
                'name': 'fp16',
                'kwargs': {}
            },
            {
                'name': 'normalized_embeddings',
                'kwargs': {}
            }
        ]
        
        for technique in optimization_techniques:
            technique_name = technique['name']
            technique_kwargs = technique['kwargs']
            
            results = optim_bench.benchmark_technique(
                technique=technique_name,
                queries=queries,
                batch_sizes=batch_sizes,
                num_runs=num_runs,
                **technique_kwargs
            )
            
            all_results.extend(results)
    
    # Save raw results
    results_file = os.path.join(run_dir, "benchmark_results.json")
    
    # Remove raw latencies from saved file to keep it smaller
    results_to_save = []
    for result in all_results:
        result_copy = result.copy()
        if 'raw_latencies' in result_copy:
            del result_copy['raw_latencies']
        results_to_save.append(result_copy)
    
    with open(results_file, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    print(f"\nRaw results saved to {results_file}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    plot_latency_histogram(all_results, plots_dir)
    plot_batch_size_comparison(all_results, plots_dir)
    plot_throughput_comparison(all_results, plots_dir)
    plot_optimization_comparison(all_results, plots_dir)
    
    # Generate report
    print("\nGenerating benchmark report...")
    report_path = generate_report(all_results, run_dir)
    
    print(f"\nBenchmark completed successfully!")
    print(f"Results directory: {run_dir}")
    print(f"Benchmark report: {report_path}")
    
    return run_dir, report_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comprehensive E5 Model Benchmark")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the local E5 model directory")
    parser.add_argument("--output_dir", type=str, default="./e5_benchmark_results",
                       help="Directory to save benchmark results")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 4, 8, 16, 32, 64],
                       help="Batch sizes to test")
    parser.add_argument("--num_runs", type=int, default=50,
                       help="Number of test runs per configuration")
    parser.add_argument("--queries_file", type=str, default=None,
                       help="Path to file with test queries")
    parser.add_argument("--baseline", action="store_true", default=True,
                       help="Run baseline implementation tests")
    parser.add_argument("--no_baseline", action="store_false", dest="baseline",
                       help="Skip baseline implementation tests")
    parser.add_argument("--optimized", action="store_true", default=True,
                       help="Run optimized implementation tests")
    parser.add_argument("--no_optimized", action="store_false", dest="optimized",
                       help="Skip optimized implementation tests")
    parser.add_argument("--optimizations", action="store_true", default=True,
                       help="Run optimization technique tests")
    parser.add_argument("--no_optimizations", action="store_false", dest="optimizations",
                       help="Skip optimization technique tests")
    
    args = parser.parse_args()
    
    run_benchmark(
        model_path=args.model_path,
        output_dir=args.output_dir,
        batch_sizes=args.batch_sizes,
        num_runs=args.num_runs,
        run_baseline=args.baseline,
        run_optimized=args.optimized,
        run_optimizations=args.optimizations,
        queries_file=args.queries_file
    )

    
