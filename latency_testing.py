import time
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import argparse
from typing import List, Dict, Any
import json
import os

class E5LatencyTest:
    def __init__(self, model_path: str, device: str = None):
        """
        Initialize the E5 model for latency testing
        
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
    
    def encode(self, text: str) -> np.ndarray:
        """
        Encode a text using the E5 model
        
        Args:
            text: Input text to encode
            
        Returns:
            Numpy array of embeddings
        """
        with torch.no_grad():
            inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            # Use the mean pooling of the last hidden state as the embedding
            embeddings = outputs.last_hidden_state.mean(dim=1)
            return embeddings.cpu().numpy()
    
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
        
        for _ in range(num_runs):
            # Randomly sample a batch of queries
            indices = np.random.choice(len(queries), batch_size)
            batch_queries = [queries[i] for i in indices]
            
            # Measure encoding time
            start_time = time.time()
            if batch_size == 1:
                self.encode(batch_queries[0])
            else:
                # For batched encoding, we tokenize together
                with torch.no_grad():
                    inputs = self.tokenizer(batch_queries, padding=True, truncation=True, return_tensors="pt").to(self.device)
                    outputs = self.model(**inputs)
            
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
            "device": self.device
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
        "How do proton pump inhibitors work to reduce stomach acid?"
    ]

def run_latency_tests(model_path: str, output_dir: str, batch_sizes: List[int] = [1, 4, 8], 
                     num_runs: int = 100, queries_file: str = None):
    """
    Run latency tests and save results
    
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
    e5_test = E5LatencyTest(model_path)
    
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
    
    # Save results
    results_df = pd.DataFrame(results)
    results_file = os.path.join(output_dir, "latency_results.csv")
    results_df.to_csv(results_file, index=False)
    print(f"\nResults saved to {results_file}")
    
    # Save detailed results as JSON
    detailed_file = os.path.join(output_dir, "latency_results.json")
    with open(detailed_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Detailed results saved to {detailed_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure latency of E5 model")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the local E5 model directory")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory to save test results")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 4, 8],
                        help="Batch sizes to test")
    parser.add_argument("--num_runs", type=int, default=100,
                        help="Number of test runs per batch size")
    parser.add_argument("--queries_file", type=str, default=None,
                        help="Path to file with test queries (JSON or text file, one query per line)")
    
    args = parser.parse_args()
    
    run_latency_tests(
        model_path=args.model_path,
        output_dir=args.output_dir,
        batch_sizes=args.batch_sizes,
        num_runs=args.num_runs,
        queries_file=args.queries_file
    )
