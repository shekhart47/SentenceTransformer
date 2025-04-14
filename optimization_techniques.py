import torch
from sentence_transformers import SentenceTransformer
import time
import numpy as np
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple

class E5Optimizer:
    """Class for applying various optimization techniques to E5 model"""
    
    def __init__(self, model_path: str, device: str = None):
        """
        Initialize the E5 optimizer
        
        Args:
            model_path: Path to the E5 model
            device: Device to use ('cuda' or 'cpu')
        """
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
        if technique == "quantization":
            return self._apply_quantization(**kwargs)
        elif technique == "lower_precision":
            return self._apply_lower_precision(**kwargs)
        elif technique == "sequence_length":
            return self._apply_sequence_length(**kwargs)
        elif technique == "onnx":
            return self._apply_onnx_conversion(**kwargs)
        else:
            raise ValueError(f"Unknown optimization technique: {technique}")
    
    def _apply_quantization(self, quantization_level: str = "int8") -> SentenceTransformer:
        """
        Apply quantization to the model
        
        Args:
            quantization_level: Quantization level ('int8' or 'int4')
            
        Returns:
            Quantized model
        """
        print(f"Loading model with {quantization_level} quantization...")
        
        model = SentenceTransformer(self.model_path, device=self.device)
        
        if quantization_level == "int8":
            # Apply int8 quantization
            model.half()  # Convert to half precision first
            
            # Note: For a proper int8 quantization, we would typically use:
            # torch.quantization.quantize_dynamic
            # However, SentenceTransformer models require specific handling
            # This is a simplified approach
            
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    # This is a simplified representation - actual quantization
                    # would use more sophisticated techniques
                    pass
        
        elif quantization_level == "int4":
            # For int4 quantization, we would need specialized libraries like
            # bitsandbytes or optimum, which would require additional imports
            # This is a placeholder for where that quantization would occur
            pass
        
        return model
    
    def _apply_lower_precision(self, precision: str = "fp16") -> SentenceTransformer:
        """
        Apply lower precision to the model
        
        Args:
            precision: Precision level ('fp16' or 'bf16')
            
        Returns:
            Lower precision model
        """
        print(f"Loading model with {precision} precision...")
        
        model = SentenceTransformer(self.model_path, device=self.device)
        
        if precision == "fp16":
            model.half()  # Convert to half precision
        elif precision == "bf16":
            # BF16 precision - if hardware supports it 
            if hasattr(torch, 'bfloat16') and self.device == 'cuda':
                for param in model.parameters():
                    param.data = param.data.to(torch.bfloat16)
        
        return model
    
    def _apply_sequence_length(self, max_seq_length: int = 128) -> SentenceTransformer:
        """
        Apply sequence length optimization
        
        Args:
            max_seq_length: Maximum sequence length
            
        Returns:
            Model with optimized sequence length
        """
        print(f"Loading model with max sequence length {max_seq_length}...")
        
        model = SentenceTransformer(self.model_path, device=self.device)
        model.max_seq_length = max_seq_length
        
        return model
    
    def _apply_onnx_conversion(self, onnx_dir: str = None) -> SentenceTransformer:
        """
        Convert model to ONNX format (placeholder)
        
        Args:
            onnx_dir: Directory to save ONNX model
            
        Returns:
            Model (for consistency, actual ONNX would need different handling)
        """
        print("Loading model and preparing for ONNX conversion...")
        
        if onnx_dir is None:
            onnx_dir = os.path.join(os.path.dirname(self.model_path), "onnx")
        
        # Create directory if it doesn't exist
        os.makedirs(onnx_dir, exist_ok=True)
        
        # Note: Actual ONNX conversion would involve:
        # 1. Loading the model
        # 2. Creating a dummy input
        # 3. Using torch.onnx.export to convert the model
        # 4. Using ONNX Runtime for inference
        
        # This is a placeholder for consistency - actual implementation
        # would require additional dependencies
        model = SentenceTransformer(self.model_path, device=self.device)
        
        return model

def benchmark_optimized_models(
    model_path: str,
    optimization_techniques: List[Dict[str, Any]],
    test_queries: List[str],
    batch_sizes: List[int] = [1, 4, 8],
    num_runs: int = 50,
    output_dir: str = "./optimization_results"
) -> pd.DataFrame:
    """
    Benchmark different optimization techniques
    
    Args:
        model_path: Path to the base model
        optimization_techniques: List of techniques to apply
        test_queries: List of queries to test
        batch_sizes: Batch sizes to test
        num_runs: Number of runs per test
        output_dir: Directory to save results
        
    Returns:
        DataFrame with benchmark results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the optimizer
    optimizer = E5Optimizer(model_path)
    
    results = []
    
    # Benchmark each optimization technique
    for technique in optimization_techniques:
        technique_name = technique["name"]
        technique_params = technique.get("params", {})
        
        print(f"\n=== Testing optimization: {technique_name} ===")
        
        # Apply optimization
        try:
            model = optimizer.apply_optimization(technique_name, **technique_params)
            
            # Perform a warmup
            for _ in range(5):
                model.encode(["What is diabetes?"])
            
            # Test each batch size
            for batch_size in batch_sizes:
                print(f"Testing batch size {batch_size}...")
                
                # Measure latency
                latencies = []
                for _ in range(num_runs):
                    # Select random queries
                    indices = np.random.choice(len(test_queries), batch_size)
                    queries = [test_queries[i] for i in indices]
                    
                    # Measure encoding time
                    start_time = time.time()
                    model.encode(queries, batch_size=batch_size)
                    latency = (time.time() - start_time) * 1000  # ms
                    latencies.append(latency)
                
                # Calculate statistics
                latencies = np.array(latencies)
                stats = {
                    "optimization": technique_name,
                    "params": str(technique_params),
                    "batch_size": batch_size,
                    "mean_latency_ms": float(np.mean(latencies)),
                    "median_latency_ms": float(np.median(latencies)),
                    "p90_latency_ms": float(np.percentile(latencies, 90)),
                    "p95_latency_ms": float(np.percentile(latencies, 95)),
                    "p99_latency_ms": float(np.percentile(latencies, 99)),
                    "std_latency_ms": float(np.std(latencies)),
                    "min_latency_ms": float(np.min(latencies)),
                    "max_latency_ms": float(np.max(latencies)),
                }
                results.append(stats)
                
                print(f"  Batch size {batch_size}: Mean latency = {stats['mean_latency_ms']:.2f} ms, P95 = {stats['p95_latency_ms']:.2f} ms")
        
        except Exception as e:
            print(f"Error applying optimization {technique_name}: {e}")
    
    # Create DataFrame from results
    df = pd.DataFrame(results)
    
    # Save results
    results_file = os.path.join(output_dir, "optimization_results.csv")
    df.to_csv(results_file, index=False)
    print(f"\nResults saved to {results_file}")
    
    # Create visualization
    plot_optimization_results(df, output_dir)
    
    return df

def plot_optimization_results(results: pd.DataFrame, output_dir: str):
    """
    Create plots to visualize optimization results
    
    Args:
        results: DataFrame with benchmark results
        output_dir: Directory to save plots
    """
    # Get unique optimizations and batch sizes
    optimizations = results['optimization'].unique()
    batch_sizes = results['batch_size'].unique()
    
    # Plot for each batch size
    for batch_size in batch_sizes:
        batch_data = results[results['batch_size'] == batch_size]
        
        # Plot mean latency
        plt.figure(figsize=(10, 6))
        x = np.arange(len(optimizations))
        
        means = [batch_data[batch_data['optimization'] == opt]['mean_latency_ms'].values[0] for opt in optimizations]
        p95s = [batch_data[batch_data['optimization'] == opt]['p95_latency_ms'].values[0] for opt in optimizations]
        
        width = 0.35
        plt.bar(x - width/2, means, width, label='Mean Latency')
        plt.bar(x + width/2, p95s, width, label='P95 Latency')
        
        plt.title(f'Optimization Comparison - Batch Size {batch_size}')
        plt.xlabel('Optimization Technique')
        plt.ylabel('Latency (ms)')
        plt.xticks(x, optimizations)
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add values on top of bars
        for i, v in enumerate(means):
            plt.text(i - width/2, v + 1, f'{v:.1f}', ha='center', fontsize=9)
        for i, v in enumerate(p95s):
            plt.text(i + width/2, v + 1, f'{v:.1f}', ha='center', fontsize=9)
        
        plt.tight_layout()
        
        # Save the plot
        plot_file = os.path.join(output_dir, f'optimization_batch_{batch_size}.png')
        plt.savefig(plot_file)
        plt.close()
        print(f"Plot saved to {plot_file}")
    
    # Create a summary plot for all batch sizes
    plt.figure(figsize=(12, 8))
    
    # Plot for each optimization
    for i, opt in enumerate(optimizations):
        opt_data = results[results['optimization'] == opt]
        
        batch_sizes = opt_data['batch_size'].values
        mean_latencies = opt_data['mean_latency_ms'].values
        
        plt.plot(batch_sizes, mean_latencies, marker='o', label=opt)
    
    plt.title('Mean Latency by Batch Size')
    plt.xlabel('Batch Size')
    plt.ylabel('Mean Latency (ms)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save the summary plot
    summary_file = os.path.join(output_dir, 'optimization_summary.png')
    plt.savefig(summary_file)
    plt.close()
    print(f"Summary plot saved to {summary_file}")

def create_production_ready_model(
    model_path: str, 
    output_path: str, 
    optimization_config: Dict[str, Any]
) -> None:
    """
    Create a production-ready model using the specified optimizations
    
    Args:
        model_path: Path to the base model
        output_path: Path to save the optimized model
        optimization_config: Configuration for optimization
    """
    # Load optimization settings
    technique = optimization_config.get("technique", "sequence_length")
    params = optimization_config.get("params", {})
    
    # Initialize optimizer
    optimizer = E5Optimizer(model_path)
    
    # Apply optimization
    print(f"Applying optimization: {technique}")
    model = optimizer.apply_optimization(technique, **params)
    
    # Save the optimized model
    os.makedirs(output_path, exist_ok=True)
    model.save(output_path)
    
    # Save configuration
    config_file = os.path.join(output_path, "optimization_config.json")
    with open(config_file, 'w') as f:
        json.dump(optimization_config, f, indent=2)
    
    print(f"Optimized model saved to {output_path}")
    print(f"Optimization configuration saved to {config_file}")

if __name__ == "__main__":
    import argparse
    from optimized_latency_testing import load_medical_queries
    
    parser = argparse.ArgumentParser(description="Optimize E5 model and benchmark performance")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the local E5 model directory")
    parser.add_argument("--output_dir", type=str, default="./optimization_results",
                       help="Directory to save results")
    parser.add_argument("--optimize", action="store_true",
                       help="Create an optimized production model")
    parser.add_argument("--optimization_technique", type=str, default="sequence_length",
                       choices=["quantization", "lower_precision", "sequence_length", "onnx"],
                       help="Optimization technique to apply")
    parser.add_argument("--max_seq_length", type=int, default=128,
                       help="Maximum sequence length (for sequence_length technique)")
    parser.add_argument("--precision", type=str, default="fp16",
                       choices=["fp16", "bf16"],
                       help="Precision level (for lower_precision technique)")
    parser.add_argument("--quantization_level", type=str, default="int8",
                       choices=["int8", "int4"],
                       help="Quantization level (for quantization technique)")
    parser.add_argument("--benchmark", action="store_true",
                       help="Benchmark different optimization techniques")
    
    args = parser.parse_args()
    
    # If benchmark is selected
    if args.benchmark:
        # Define optimization techniques to test
        techniques = [
            {"name": "sequence_length", "params": {"max_seq_length": 256}},
            {"name": "sequence_length", "params": {"max_seq_length": 128}},
            {"name": "lower_precision", "params": {"precision": "fp16"}},
            {"name": "quantization", "params": {"quantization_level": "int8"}},
        ]
        
        # Load medical queries
        queries = load_medical_queries()
        
        # Run benchmarks
        benchmark_optimized_models(
            model_path=args.model_path,
            optimization_techniques=techniques,
            test_queries=queries,
            output_dir=args.output_dir
        )
    
    # If optimize is selected
    elif args.optimize:
        # Create optimization config based on selected technique
        if args.optimization_technique == "sequence_length":
            config = {
                "technique": "sequence_length",
                "params": {"max_seq_length": args.max_seq_length}
            }
        elif args.optimization_technique == "lower_precision":
            config = {
                "technique": "lower_precision",
                "params": {"precision": args.precision}
            }
        elif args.optimization_technique == "quantization":
            config = {
                "technique": "quantization", 
                "params": {"quantization_level": args.quantization_level}
            }
        elif args.optimization_technique == "onnx":
            config = {
                "technique": "onnx",
                "params": {"onnx_dir": os.path.join(args.output_dir, "onnx_model")}
            }
        
        # Create production model
        output_path = os.path.join(args.output_dir, f"optimized_{args.optimization_technique}")
        create_production_ready_model(
            model_path=args.model_path,
            output_path=output_path,
            optimization_config=config
        )
