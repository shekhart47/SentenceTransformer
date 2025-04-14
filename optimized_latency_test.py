import time
import argparse
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import pandas as pd
import json
import os

class OptimizedE5LatencyTester:
    def __init__(self, model_path: str, device: str = None):
        """
        Initialize the optimized E5 model using SentenceTransformers for latency testing.
        
        Args:
            model_path: Path to the local E5 model
            device: Device to run the model on ('cpu', 'cuda', 'mps', etc.)
        """
        self.model_path = model_path
        
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading optimized E5 model from {model_path} on {self.device}")
        
        # Load model with SentenceTransformers
        # When loading from a local path, we need to pass it as a model path
        self.model = SentenceTransformer(model_path, device=self.device)
        
        # Set common parameters that affect performance
        self.model.max_seq_length = 512  # You can adjust based on your needs
        
        # Warm up the model
        print("Warming up model...")
        self._warmup()
        
    def _warmup(self, num_warmup: int = 5):
        """Warm up the model with a few inferences to initialize cache"""
        sample_text = "This is a sample text for model warmup"
        for _ in range(num_warmup):
            self.model.encode(sample_text)
    
    def encode(self, text: str, batch_size: int = 1) -> np.ndarray:
        """
        Encode a single text using the E5 optimized model.
        
        Args:
            text: Text to encode
            batch_size: Batch size for encoding
            
        Returns:
            Embedding as numpy array
        """
        # Prepare the input text with the instruction prefix for E5
        if not text.startswith(("query: ", "passage: ")):
            if len(text.strip().split()) <= 10:  # Heuristic to determine if it's a query
                text = f"query: {text}"
            else:
                text = f"passage: {text}"
        
        # SentenceTransformers handles tokenization, encoding, pooling internally
        embedding = self.model.encode(text, batch_size=batch_size, 
                                     normalize_embeddings=True, 
                                     show_progress_bar=False)
        
        return embedding
    
    def measure_latency(self, queries: List[str], n_runs: int = 5, batch_size: int = 1) -> Dict:
        """
        Measure latency statistics for a list of queries.
        
        Args:
            queries: List of query strings to encode
            n_runs: Number of times to run each query for statistical significance
            batch_size: Batch size for encoding (default: 1 for per-query latency)
            
        Returns:
            Dictionary with latency statistics
        """
        all_latencies = []
        query_results = {}
        
        # Test single query latency
        for query in queries:
            query_latencies = []
            
            # Run n times for each query
            for _ in range(n_runs):
                start_time = time.time()
                _ = self.encode(query, batch_size=batch_size)
                end_time = time.time()
                
                latency = (end_time - start_time) * 1000  # Convert to ms
                query_latencies.append(latency)
                all_latencies.append(latency)
            
            # Calculate stats for this query
            query_results[query] = {
                "mean": np.mean(query_latencies),
                "min": np.min(query_latencies),
                "max": np.max(query_latencies),
                "std": np.std(query_latencies),
                "raw": query_latencies
            }
        
        # Calculate overall percentiles
        percentiles = {
            "p50": np.percentile(all_latencies, 50),
            "p90": np.percentile(all_latencies, 90),
            "p95": np.percentile(all_latencies, 95),
            "p99": np.percentile(all_latencies, 99)
        }
        
        # Calculate overall stats
        overall_stats = {
            "mean": np.mean(all_latencies),
            "min": np.min(all_latencies),
            "max": np.max(all_latencies),
            "std": np.std(all_latencies),
            "count": len(all_latencies),
            "percentiles": percentiles
        }
        
        return {
            "overall": overall_stats,
            "per_query": query_results
        }
    
    def measure_batch_latency(self, queries: List[str], n_runs: int = 5, 
                             batch_sizes: List[int] = [1, 4, 8, 16, 32]) -> Dict:
        """
        Measure latency statistics for different batch sizes.
        
        Args:
            queries: List of query strings to encode
            n_runs: Number of times to run each batch for statistical significance
            batch_sizes: List of batch sizes to test
            
        Returns:
            Dictionary with batch latency statistics
        """
        batch_results = {}
        
        for batch_size in batch_sizes:
            print(f"Testing batch size: {batch_size}")
            batch_latencies = []
            
            # Prepare prefixed queries
            prefixed_queries = []
            for query in queries:
                if not query.startswith(("query: ", "passage: ")):
                    if len(query.strip().split()) <= 10:
                        prefixed_queries.append(f"query: {query}")
                    else:
                        prefixed_queries.append(f"passage: {query}")
                else:
                    prefixed_queries.append(query)
            
            # If queries are fewer than batch size, replicate to match
            if len(prefixed_queries) < batch_size:
                prefixed_queries = (prefixed_queries * (batch_size // len(prefixed_queries) + 1))[:batch_size]
            
            # Run n times for each batch size
            for _ in range(n_runs):
                start_time = time.time()
                # Use the batch encoding capability of SentenceTransformers
                _ = self.model.encode(prefixed_queries[:batch_size], batch_size=batch_size, 
                                    normalize_embeddings=True, show_progress_bar=False)
                end_time = time.time()
                
                total_time = end_time - start_time
                per_query_time = (total_time / batch_size) * 1000  # Convert to ms per query
                batch_latencies.append(per_query_time)
            
            # Calculate stats for this batch size
            batch_results[f"batch_{batch_size}"] = {
                "mean_per_query": np.mean(batch_latencies),
                "min_per_query": np.min(batch_latencies),
                "max_per_query": np.max(batch_latencies),
                "std_per_query": np.std(batch_latencies),
                "raw": batch_latencies
            }
        
        return batch_results

def load_medical_queries(query_file: str = None) -> List[str]:
    """
    Load medical queries from a file or use defaults if no file provided.
    
    Args:
        query_file: Path to a JSON or text file with medical queries
        
    Returns:
        List of medical query strings
    """
    default_queries = [
        "What are the symptoms of Type 2 Diabetes?",
        "How is hypertension diagnosed?",
        "What are common side effects of statins?",
        "Can you describe the pathophysiology of heart failure?",
        "What's the difference between CT and MRI imaging?",
        "How does chemotherapy work for cancer treatment?",
        "What are the latest treatments for rheumatoid arthritis?",
        "Explain the mechanism of action for ACE inhibitors",
        "What causes chronic kidney disease?",
        "How is multiple sclerosis diagnosed and treated?",
        "What are the risk factors for developing Alzheimer's disease?",
        "How does insulin resistance lead to Type 2 Diabetes?",
        "What are the stages of chronic obstructive pulmonary disease?",
        "Describe the pathophysiology of asthma attacks",
        "What are common complications of untreated hypothyroidism?"
    ]
    
    if query_file is None:
        return default_queries
    
    # Load queries from file if provided
    try:
        ext = os.path.splitext(query_file)[1].lower()
        if ext == '.json':
            with open(query_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict) and 'queries' in data:
                    return data['queries']
                else:
                    print(f"Warning: Couldn't parse JSON format. Using default queries.")
                    return default_queries
        else:
            # Assume text file with one query per line
            with open(query_file, 'r') as f:
                queries = [line.strip() for line in f if line.strip()]
                return queries
    except Exception as e:
        print(f"Error loading queries file: {e}")
        print("Using default medical queries instead.")
        return default_queries

def save_results(results: Dict, output_file: str = "optimized_latency_results.json"):
    """
    Save latency test results to a file.
    
    Args:
        results: Results dictionary from measure_latency
        output_file: Path to save results
    """
    # Create a copy of results that's JSON serializable
    serializable_results = {}
    
    # Process overall results if present
    if "overall" in results:
        serializable_results["overall"] = {
            "mean": float(results["overall"]["mean"]),
            "min": float(results["overall"]["min"]),
            "max": float(results["overall"]["max"]),
            "std": float(results["overall"]["std"]),
            "count": int(results["overall"]["count"]),
            "percentiles": {
                k: float(v) for k, v in results["overall"]["percentiles"].items()
            }
        }
    
    # Process per-query results if present
    if "per_query" in results:
        serializable_results["per_query"] = {}
        for query, query_data in results["per_query"].items():
            serializable_results["per_query"][query] = {
                "mean": float(query_data["mean"]),
                "min": float(query_data["min"]),
                "max": float(query_data["max"]),
                "std": float(query_data["std"]),
                "raw": [float(x) for x in query_data["raw"]]
            }
    
    # Process batch results if present
    if any(k.startswith("batch_") for k in results.keys()):
        serializable_results["batch_results"] = {}
        for batch_size, batch_data in results.items():
            if batch_size.startswith("batch_"):
                serializable_results["batch_results"][batch_size] = {
                    "mean_per_query": float(batch_data["mean_per_query"]),
                    "min_per_query": float(batch_data["min_per_query"]),
                    "max_per_query": float(batch_data["max_per_query"]),
                    "std_per_query": float(batch_data["std_per_query"]),
                    "raw": [float(x) for x in batch_data["raw"]]
                }
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Results saved to {output_file}")

def print_summary(results: Dict):
    """
    Print a summary of latency test results.
    
    Args:
        results: Results dictionary from measure_latency
    """
    print("\n" + "="*50)
    print(f"OPTIMIZED LATENCY TEST SUMMARY")
    print("="*50)
    
    # Print overall results if present
    if "overall" in results:
        overall = results["overall"]
        print(f"Total queries processed: {overall['count']}")
        print(f"Mean latency: {overall['mean']:.2f} ms")
        print(f"Min latency: {overall['min']:.2f} ms")
        print(f"Max latency: {overall['max']:.2f} ms")
        print(f"Standard deviation: {overall['std']:.2f} ms")
        print("\nPercentiles:")
        for p_name, p_value in overall["percentiles"].items():
            print(f"  {p_name}: {p_value:.2f} ms")
    
    # Print batch results if present
    if any(k.startswith("batch_") for k in results.keys()):
        print("\nBatch Processing Results (ms per query):")
        print("-"*40)
        
        batch_sizes = sorted([k for k in results.keys() if k.startswith("batch_")], 
                           key=lambda x: int(x.split("_")[1]))
        
        for batch_size in batch_sizes:
            data = results[batch_size]
            print(f"Batch size {batch_size.split('_')[1]}:")
            print(f"  Mean latency per query: {data['mean_per_query']:.2f} ms")
            print(f"  Min latency per query: {data['min_per_query']:.2f} ms")
            print(f"  Max latency per query: {data['max_per_query']:.2f} ms")
            print(f"  Std deviation: {data['std_per_query']:.2f} ms")
    
    print("="*50)

def main():
    parser = argparse.ArgumentParser(description="Optimized E5 model latency testing with SentenceTransformers")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the local E5 model")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run inference on (cpu, cuda, etc.)")
    parser.add_argument("--query_file", type=str, default=None,
                        help="Path to file with medical queries (JSON or TXT)")
    parser.add_argument("--n_runs", type=int, default=5,
                        help="Number of runs per query for statistical significance")
    parser.add_argument("--test_batching", action="store_true",
                        help="Whether to test different batch sizes")
    parser.add_argument("--batch_sizes", type=str, default="1,4,8,16,32",
                        help="Comma-separated list of batch sizes to test")
    parser.add_argument("--output_file", type=str, default="e5_optimized_latency_results.json",
                        help="Path to save results")
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = OptimizedE5LatencyTester(model_path=args.model_path, device=args.device)
    
    # Load queries
    queries = load_medical_queries(args.query_file)
    print(f"Testing latency on {len(queries)} medical queries, {args.n_runs} runs each")
    
    # Measure single query latency
    results = tester.measure_latency(queries, n_runs=args.n_runs)
    
    # Test batch processing if requested
    if args.test_batching:
        batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
        batch_results = tester.measure_batch_latency(
            queries, n_runs=args.n_runs, batch_sizes=batch_sizes)
        # Merge the results
        results.update(batch_results)
    
    # Print summary
    print_summary(results)
    
    # Save results
    save_results(results, args.output_file)

if __name__ == "__main__":
    main()
