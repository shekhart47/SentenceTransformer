import time
import argparse
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from typing import List, Dict
import pandas as pd
import json
import os

class E5LatencyTester:
    def __init__(self, model_path: str, device: str = None):
        """
        Initialize the E5 model for latency testing.
        
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
            
        print(f"Loading E5 model from {model_path} on {self.device}")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()  # Set model to evaluation mode
        
        # Warm up the model
        print("Warming up model...")
        self._warmup()
        
    def _warmup(self, num_warmup: int = 5):
        """Warm up the model with a few inferences to initialize cache"""
        with torch.no_grad():
            sample_text = "This is a sample text for model warmup"
            inputs = self.tokenizer(sample_text, return_tensors="pt").to(self.device)
            for _ in range(num_warmup):
                self.model(**inputs)
    
    def encode(self, text: str) -> np.ndarray:
        """
        Encode a single text using the E5 model.
        
        Args:
            text: Text to encode
            
        Returns:
            Embedding as numpy array
        """
        # Prepare the input text with the instruction prefix for E5
        # "passage: " prefix is for document/passage encoding
        if not text.startswith(("query: ", "passage: ")):
            if len(text.strip().split()) <= 10:  # Heuristic to determine if it's a query
                text = f"query: {text}"
            else:
                text = f"passage: {text}"
                
        # Tokenize and get embedding
        with torch.no_grad():
            inputs = self.tokenizer(text, padding=True, truncation=True, 
                                   return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            # Use mean pooling to get embedding
            embedding = self._mean_pooling(outputs, inputs['attention_mask'])
            # Normalize embedding
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
            
        return embedding.cpu().numpy()
    
    def _mean_pooling(self, model_output, attention_mask):
        """
        Mean pooling operation to get sentence embeddings
        
        Args:
            model_output: Output from the model
            attention_mask: Attention mask from tokenizer
            
        Returns:
            Pooled embedding tensor
        """
        token_embeddings = model_output[0]  # First element contains token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def measure_latency(self, queries: List[str], n_runs: int = 5) -> Dict:
        """
        Measure latency statistics for a list of queries.
        
        Args:
            queries: List of query strings to encode
            n_runs: Number of times to run each query for statistical significance
            
        Returns:
            Dictionary with latency statistics
        """
        all_latencies = []
        query_results = {}
        
        for query in queries:
            query_latencies = []
            
            # Run n times for each query
            for _ in range(n_runs):
                start_time = time.time()
                _ = self.encode(query)
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

def save_results(results: Dict, output_file: str = "latency_results.json"):
    """
    Save latency test results to a file.
    
    Args:
        results: Results dictionary from measure_latency
        output_file: Path to save results
    """
    # Create a copy of results that's JSON serializable
    serializable_results = {
        "overall": {
            "mean": float(results["overall"]["mean"]),
            "min": float(results["overall"]["min"]),
            "max": float(results["overall"]["max"]),
            "std": float(results["overall"]["std"]),
            "count": int(results["overall"]["count"]),
            "percentiles": {
                k: float(v) for k, v in results["overall"]["percentiles"].items()
            }
        },
        "per_query": {}
    }
    
    # Process per-query results
    for query, query_data in results["per_query"].items():
        serializable_results["per_query"][query] = {
            "mean": float(query_data["mean"]),
            "min": float(query_data["min"]),
            "max": float(query_data["max"]),
            "std": float(query_data["std"]),
            "raw": [float(x) for x in query_data["raw"]]
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
    overall = results["overall"]
    
    print("\n" + "="*50)
    print(f"LATENCY TEST SUMMARY")
    print("="*50)
    print(f"Total queries processed: {overall['count']}")
    print(f"Mean latency: {overall['mean']:.2f} ms")
    print(f"Min latency: {overall['min']:.2f} ms")
    print(f"Max latency: {overall['max']:.2f} ms")
    print(f"Standard deviation: {overall['std']:.2f} ms")
    print("\nPercentiles:")
    for p_name, p_value in overall["percentiles"].items():
        print(f"  {p_name}: {p_value:.2f} ms")
    print("="*50)

def main():
    parser = argparse.ArgumentParser(description="E5 model latency testing")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the local E5 model")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run inference on (cpu, cuda, etc.)")
    parser.add_argument("--query_file", type=str, default=None,
                        help="Path to file with medical queries (JSON or TXT)")
    parser.add_argument("--n_runs", type=int, default=5,
                        help="Number of runs per query for statistical significance")
    parser.add_argument("--output_file", type=str, default="e5_base_latency_results.json",
                        help="Path to save results")
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = E5LatencyTester(model_path=args.model_path, device=args.device)
    
    # Load queries
    queries = load_medical_queries(args.query_file)
    print(f"Testing latency on {len(queries)} medical queries, {args.n_runs} runs each")
    
    # Measure latency
    results = tester.measure_latency(queries, n_runs=args.n_runs)
    
    # Print summary
    print_summary(results)
    
    # Save results
    save_results(results, args.output_file)

if __name__ == "__main__":
    main()
