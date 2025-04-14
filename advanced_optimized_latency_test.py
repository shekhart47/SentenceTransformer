import argparse
import json
import os
import time
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Union
from sentence_transformers import SentenceTransformer
import onnxruntime as ort
from transformers import AutoTokenizer
from tqdm import tqdm

class AdvancedE5Optimizer:
    """
    Class for implementing advanced optimization techniques on E5 models
    to further reduce inference latency beyond SentenceTransformers.
    """
    
    def __init__(self, model_path: str, cache_dir: str = "./optimized_models"):
        """
        Initialize the advanced optimizer.
        
        Args:
            model_path: Path to the local E5 model
            cache_dir: Directory to save optimized models
        """
        self.model_path = model_path
        self.cache_dir = cache_dir
        self.model_name = os.path.basename(os.path.normpath(model_path))
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load tokenizer (will be used across different optimization methods)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Base SentenceTransformer model (reference implementation)
        self.st_model = None
        
        # Initialize containers for optimized models
        self.fp16_model = None
        self.int8_model = None
        self.onnx_model = None
        self.onnx_session = None
    
    def _format_query(self, text: str) -> str:
        """Format the input text with the appropriate prefix for E5."""
        if not text.startswith(("query: ", "passage: ")):
            if len(text.strip().split()) <= 10:  # Heuristic to determine if it's a query
                text = f"query: {text}"
            else:
                text = f"passage: {text}"
        return text
    
    def load_base_model(self, device: str = None) -> SentenceTransformer:
        """
        Load the base SentenceTransformer model for comparison.
        
        Args:
            device: Device to load the model on (defaults to CUDA if available)
            
        Returns:
            Loaded SentenceTransformer model
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        print(f"Loading base SentenceTransformer model from {self.model_path} on {device}")
        self.st_model = SentenceTransformer(self.model_path, device=device)
        return self.st_model
    
    def optimize_fp16(self, device: str = None) -> SentenceTransformer:
        """
        Convert the model to FP16 precision for faster inference.
        
        Args:
            device: Device to load the model on (defaults to CUDA if available)
            
        Returns:
            FP16 optimized model
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        if device == "cpu":
            print("Warning: FP16 optimization works best with CUDA devices")
        
        print(f"Creating FP16 optimized model on {device}")
        
        # Load the model if not already loaded
        if self.st_model is None:
            self.load_base_model(device)
        
        # Convert to half precision (FP16)
        self.fp16_model = self.st_model.half()
        
        return self.fp16_model
    
    def optimize_int8(self, device: str = None) -> SentenceTransformer:
        """
        Quantize the model to INT8 precision using PyTorch quantization.
        
        Args:
            device: Device to load the model on
            
        Returns:
            INT8 quantized model
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Creating INT8 quantized model on {device}")
        
        # Load the model if not already loaded
        if self.st_model is None:
            self.load_base_model(device)
        
        # For PyTorch 2.0+, we can use quantization APIs
        if hasattr(torch, "quantization") and hasattr(torch.quantization, "quantize_dynamic"):
            # We need to move the model to CPU for quantization
            orig_device = self.st_model._target_device
            
            # Move model to CPU for quantization
            self.st_model.to('cpu')
            
            # Get the model
            model = self.st_model._modules['0']
            
            # Perform dynamic quantization
            quantized_model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
            
            # Replace the transformer model with the quantized version
            self.int8_model = self.st_model
            self.int8_model._modules['0'] = quantized_model
            
            # Move back to original device
            self.int8_model.to(orig_device)
        else:
            print("Warning: PyTorch quantization APIs not available. Skipping INT8 quantization.")
            self.int8_model = self.st_model
        
        return self.int8_model
    
    def optimize_onnx(self, 
                     output_path: str = None, 
                     opset_version: int = 14,
                     device: str = None,
                     use_gpu: bool = None,
                     dynamic_axes: bool = True,
                     optimize_level: int = 99) -> str:
        """
        Convert the model to ONNX format and optimize for inference.
        
        Args:
            output_path: Path to save the ONNX model
            opset_version: ONNX opset version to use
            device: Device to load the PyTorch model on
            use_gpu: Whether to use GPU for ONNX runtime (defaults to use if available)
            dynamic_axes: Whether to use dynamic axes for variable input lengths
            optimize_level: ONNX optimization level
            
        Returns:
            Path to the optimized ONNX model
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        if use_gpu is None:
            use_gpu = torch.cuda.is_available()
        
        # Define output path if not provided
        if output_path is None:
            output_path = os.path.join(self.cache_dir, f"{self.model_name}_optimized.onnx")
        
        # Check if model already exists
        if os.path.exists(output_path):
            print(f"ONNX model already exists at {output_path}. Loading...")
            self.onnx_model = output_path
            
            # Create ONNX Runtime session
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
            self.onnx_session = ort.InferenceSession(output_path, providers=providers)
            
            return output_path
        
        print(f"Creating ONNX optimized model at {output_path}")
        
        # Load the model if not already loaded
        if self.st_model is None:
            self.load_base_model(device)
        
        # Prepare example inputs for tracing
        sample_text = "This is a sample text for ONNX conversion"
        sample_text = self._format_query(sample_text)
        
        # Tokenize
        inputs = self.tokenizer(
            sample_text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)
        
        # Set up dynamic axes if requested
        dynamic_axes_dict = None
        if dynamic_axes:
            dynamic_axes_dict = {
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
                'token_type_ids': {0: 'batch_size', 1: 'sequence_length'} if 'token_type_ids' in inputs else None,
                'outputs': {0: 'batch_size'}
            }
            # Remove None entries
            dynamic_axes_dict = {k: v for k, v in dynamic_axes_dict.items() if v is not None}
        
        # Get the transformer model
        model = self.st_model._modules['0']
        model.eval()
        
        # Export to ONNX
        with torch.no_grad():
            torch.onnx.export(
                model,
                (inputs.input_ids, inputs.attention_mask),
                output_path,
                opset_version=opset_version,
                input_names=['input_ids', 'attention_mask'],
                output_names=['outputs'],
                dynamic_axes=dynamic_axes_dict,
                export_params=True,
                do_constant_folding=True
            )
        
        # Optimize the ONNX model
        print("Optimizing ONNX model...")
        import onnxruntime.transformers.optimizer as optimizer
        
        optimized_model = optimizer.optimize_model(
            output_path,
            model_type='bert',  # E5 is a BERT-based model
            num_heads=12,  # Typically 12 for base models, adjust for large/xlarge
            hidden_size=768,  # Adjust based on model size
            optimization_level=optimize_level
        )
        
        # Save optimized model
        optimized_model.save_model_to_file(output_path)
        
        # Store the model path
        self.onnx_model = output_path
        
        # Create ONNX Runtime session
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
        self.onnx_session = ort.InferenceSession(output_path, providers=providers)
        
        return output_path
    
    def encode_base(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """
        Encode texts using the base SentenceTransformer model.
        
        Args:
            texts: Text or list of texts to encode
            batch_size: Batch size for encoding
            
        Returns:
            Numpy array of embeddings
        """
        if self.st_model is None:
            self.load_base_model()
        
        # Format queries if needed
        if isinstance(texts, str):
            texts = [self._format_query(texts)]
        else:
            texts = [self._format_query(text) for text in texts]
        
        # Encode
        embeddings = self.st_model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        return embeddings
    
    def encode_fp16(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """
        Encode texts using the FP16 optimized model.
        
        Args:
            texts: Text or list of texts to encode
            batch_size: Batch size for encoding
            
        Returns:
            Numpy array of embeddings
        """
        if self.fp16_model is None:
            self.optimize_fp16()
        
        # Format queries if needed
        if isinstance(texts, str):
            texts = [self._format_query(texts)]
        else:
            texts = [self._format_query(text) for text in texts]
        
        # Encode with FP16 model
        embeddings = self.fp16_model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        return embeddings
    
    def encode_int8(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """
        Encode texts using the INT8 quantized model.
        
        Args:
            texts: Text or list of texts to encode
            batch_size: Batch size for encoding
            
        Returns:
            Numpy array of embeddings
        """
        if self.int8_model is None:
            self.optimize_int8()
        
        # Format queries if needed
        if isinstance(texts, str):
            texts = [self._format_query(texts)]
        else:
            texts = [self._format_query(text) for text in texts]
        
        # Encode with INT8 model
        embeddings = self.int8_model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        return embeddings
    
    def encode_onnx(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """
        Encode texts using the ONNX optimized model.
        
        Args:
            texts: Text or list of texts to encode
            batch_size: Batch size for encoding
            
        Returns:
            Numpy array of embeddings
        """
        if self.onnx_session is None:
            self.optimize_onnx()
        
        # Format queries if needed
        if isinstance(texts, str):
            texts = [self._format_query(texts)]
        else:
            texts = [self._format_query(text) for text in texts]
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # Run inference
            ort_inputs = {
                'input_ids': inputs['input_ids'].numpy(),
                'attention_mask': inputs['attention_mask'].numpy()
            }
            
            # Get model outputs
            outputs = self.onnx_session.run(None, ort_inputs)
            token_embeddings = outputs[0]
            
            # Apply mean pooling (similar to SentenceTransformers)
            attention_mask = inputs['attention_mask'].numpy()
            input_mask_expanded = np.expand_dims(attention_mask, axis=-1)
            
            # Mean pooling
            sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
            sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
            embeddings = sum_embeddings / sum_mask
            
            # Normalize
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            all_embeddings.append(embeddings)
        
        # Combine all batches
        if len(all_embeddings) > 0:
            return np.vstack(all_embeddings)
        else:
            return np.array([])
            
    def measure_latency(self, queries: List[str], methods: List[str], n_runs: int = 5, 
                       batch_size: int = 32) -> Dict:
        """
        Measure latency statistics for different optimization methods.
        
        Args:
            queries: List of query strings to encode
            methods: List of optimization methods to test ('base', 'fp16', 'int8', 'onnx')
            n_runs: Number of times to run each query for statistical significance
            batch_size: Batch size for encoding
            
        Returns:
            Dictionary with latency statistics for each method
        """
        results = {}
        
        # Validate methods
        valid_methods = ['base', 'fp16', 'int8', 'onnx']
        methods = [m for m in methods if m in valid_methods]
        
        for method in methods:
            print(f"Testing {method} optimization method...")
            
            # Initialize model if needed
            if method == 'base' and self.st_model is None:
                self.load_base_model()
            elif method == 'fp16' and self.fp16_model is None:
                self.optimize_fp16()
            elif method == 'int8' and self.int8_model is None:
                self.optimize_int8()
            elif method == 'onnx' and self.onnx_session is None:
                self.optimize_onnx()
            
            encode_func = getattr(self, f"encode_{method}")
            method_results = self._test_latency(queries, encode_func, n_runs, batch_size)
            results[method] = method_results
        
        return results
    
    def _test_latency(self, queries: List[str], encode_func, n_runs: int, batch_size: int) -> Dict:
        """
        Test latency for a specific encoding function.
        
        Args:
            queries: List of query strings to encode
            encode_func: Function to use for encoding
            n_runs: Number of times to run each query
            batch_size: Batch size for encoding
            
        Returns:
            Dictionary with latency statistics
        """
        # Single query latency
        single_latencies = []
        
        print("Testing single query latency...")
        for query in tqdm(queries):
            for _ in range(n_runs):
                start_time = time.time()
                _ = encode_func(query)
                end_time = time.time()
                
                latency = (end_time - start_time) * 1000  # Convert to ms
                single_latencies.append(latency)
        
        # Batch latency
        batch_latencies = []
        
        print("Testing batch latency...")
        for _ in range(n_runs):
            start_time = time.time()
            _ = encode_func(queries, batch_size=batch_size)
            end_time = time.time()
            
            total_time = (end_time - start_time) * 1000  # Convert to ms
            per_query_time = total_time / len(queries)
            batch_latencies.append(per_query_time)
        
        # Calculate statistics
        single_percentiles = {
            "p50": np.percentile(single_latencies, 50),
            "p90": np.percentile(single_latencies, 90),
            "p95": np.percentile(single_latencies, 95),
            "p99": np.percentile(single_latencies, 99)
        }
        
        batch_percentiles = {
            "p50": np.percentile(batch_latencies, 50),
            "p90": np.percentile(batch_latencies, 90),
            "p95": np.percentile(batch_latencies, 95),
            "p99": np.percentile(batch_latencies, 99)
        }
        
        return {
            "single_query": {
                "mean": np.mean(single_latencies),
                "min": np.min(single_latencies),
                "max": np.max(single_latencies),
                "std": np.std(single_latencies),
                "count": len(single_latencies),
                "percentiles": single_percentiles
            },
            "batch": {
                "mean": np.mean(batch_latencies),
                "min": np.min(batch_latencies),
                "max": np.max(batch_latencies),
                "std": np.std(batch_latencies),
                "count": len(batch_latencies),
                "percentiles": batch_percentiles
            }
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

def save_results(results: Dict, output_file: str = "advanced_optimization_results.json"):
    """
    Save latency test results to a file.
    
    Args:
        results: Results dictionary from measure_latency
        output_file: Path to save results
    """
    # Create a serializable copy of the results
    serializable_results = {}
    
    for method, method_data in results.items():
        serializable_results[method] = {}
        
        for test_type, test_data in method_data.items():
            serializable_results[method][test_type] = {
                "mean": float(test_data["mean"]),
                "min": float(test_data["min"]),
                "max": float(test_data["max"]),
                "std": float(test_data["std"]),
                "count": int(test_data["count"]),
                "percentiles": {
                    k: float(v) for k, v in test_data["percentiles"].items()
                }
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
    print("\n" + "="*60)
    print(f"ADVANCED OPTIMIZATION LATENCY TEST SUMMARY")
    print("="*60)
    
    methods = list(results.keys())
    
    if 'base' in methods:
        base_single_mean = results['base']['single_query']['mean']
        base_batch_mean = results['base']['batch']['mean']
    else:
        base_single_mean = None
        base_batch_mean = None
    
    # Print results for each method
    for method in methods:
        method_data = results[method]
        
        print(f"\n{method.upper()} OPTIMIZATION:")
        print("-" * 40)
        
        # Single query results
        single_data = method_data['single_query']
        print(f"Single Query Latency:")
        print(f"  Mean: {single_data['mean']:.2f} ms")
        print(f"  P95: {single_data['percentiles']['p95']:.2f} ms")
        
        # Print improvement over base if available
        if base_single_mean is not None and method != 'base':
            improvement = ((base_single_mean - single_data['mean']) / base_single_mean) * 100
            print(f"  Improvement over base: {improvement:.2f}%")
        
        # Batch results
        batch_data = method_data['batch']
        print(f"Batch Processing Latency (per query):")
        print(f"  Mean: {batch_data['mean']:.2f} ms")
        print(f"  P95: {batch_data['percentiles']['p95']:.2f} ms")
        
        # Print improvement over base if available
        if base_batch_mean is not None and method != 'base':
            improvement = ((base_batch_mean - batch_data['mean']) / base_batch_mean) * 100
            print(f"  Improvement over base: {improvement:.2f}%")
    
    print("\nPercentile Comparison (P95, single query):")
    for method in methods:
        p95 = results[method]['single_query']['percentiles']['p95']
        print(f"  {method.upper()}: {p95:.2f} ms")
    
    print("\nPercentile Comparison (P95, batch processing):")
    for method in methods:
        p95 = results[method]['batch']['percentiles']['p95']
        print(f"  {method.upper()}: {p95:.2f} ms")
    
    print("="*60)

def verify_model_correctness(optimizer: AdvancedE5Optimizer, queries: List[str], tolerance: float = 0.1):
    """
    Verify that optimized models produce similar results to the base model.
    
    Args:
        optimizer: Initialized AdvancedE5Optimizer
        queries: List of queries to test
        tolerance: Cosine similarity tolerance (lower values = stricter comparison)
    """
    print("\nVerifying model correctness...")
    
    # Ensure base model is loaded
    if optimizer.st_model is None:
        optimizer.load_base_model()
    
    # Get base model embeddings
    base_embeddings = optimizer.encode_base(queries)
    
    # Test each optimization method
    methods_to_test = []
    
    if optimizer.fp16_model is not None:
        methods_to_test.append(('fp16', optimizer.encode_fp16))
    
    if optimizer.int8_model is not None:
        methods_to_test.append(('int8', optimizer.encode_int8))
    
    if optimizer.onnx_session is not None:
        methods_to_test.append(('onnx', optimizer.encode_onnx))
    
    # Calculate cosine similarity between base and optimized embeddings
    for method_name, encode_func in methods_to_test:
        print(f"Testing {method_name} model correctness...")
        
        try:
            optimized_embeddings = encode_func(queries)
            
            # Check shape
            if optimized_embeddings.shape != base_embeddings.shape:
                print(f"Warning: {method_name} embeddings shape {optimized_embeddings.shape} " +
                     f"differs from base shape {base_embeddings.shape}")
                continue
            
            # Calculate cosine similarity for each embedding
            similarities = []
            for i in range(len(queries)):
                base_emb = base_embeddings[i]
                opt_emb = optimized_embeddings[i]
                
                # Normalize (in case they aren't already)
                base_emb = base_emb / np.linalg.norm(base_emb)
                opt_emb = opt_emb / np.linalg.norm(opt_emb)
                
                # Calculate cosine similarity
                similarity = np.dot(base_emb, opt_emb)
                similarities.append(similarity)
            
            # Calculate statistics
            mean_sim = np.mean(similarities)
            min_sim = np.min(similarities)
            
            print(f"  Mean similarity: {mean_sim:.4f}")
            print(f"  Min similarity: {min_sim:.4f}")
            
            if min_sim < (1.0 - tolerance):
                print(f"  Warning: Some embeddings differ significantly from base model. Min similarity: {min_sim:.4f}")
            else:
                print(f"  All embeddings are within tolerance ({tolerance}) of base model.")
                
        except Exception as e:
            print(f"  Error testing {method_name} model: {e}")

def main():
    parser = argparse.ArgumentParser(description="Advanced E5 model optimizations")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the local E5 model")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run inference on (cpu, cuda, etc.)")
    parser.add_argument("--query_file", type=str, default=None,
                        help="Path to file with medical queries (JSON or TXT)")
    parser.add_argument("--n_runs", type=int, default=5,
                        help="Number of runs per query for statistical significance")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for batch processing")
    parser.add_argument("--output_file", type=str, default="advanced_optimization_results.json",
                        help="Path to save results")
    parser.add_argument("--cache_dir", type=str, default="./optimized_models",
                        help="Directory to save optimized models")
    parser.add_argument("--optimizations", type=str, default="fp16,int8,onnx",
                        help="Comma-separated list of optimizations to test")
    parser.add_argument("--verify", action="store_true",
                        help="Verify that optimized models produce similar results to base model")
    
    args = parser.parse_args()
    
    # Parse optimizations
    optimizations = ["base"] + [opt.strip() for opt in args.optimizations.split(",")]
    
    # Load queries
    queries = load_medical_queries(args.query_file)
    print(f"Testing latency on {len(queries)} medical queries, {args.n_runs} runs each")
    
    # Initialize optimizer
    optimizer = AdvancedE5Optimizer(args.model_path, args.cache_dir)
    
    # Ensure base model is loaded
    optimizer.load_base_model(args.device)
    
    # Load/generate optimized models
    if "fp16" in optimizations:
        optimizer.optimize_fp16(args.device)
    
    if "int8" in optimizations:
        optimizer.optimize_int8(args.device)
    
    if "onnx" in optimizations:
        optimizer.optimize_onnx(device=args.device)
    
    # Verify model correctness if requested
    if args.verify:
        verify_model_correctness(optimizer, queries[:5])  # Use a subset for speed
    
    # Measure latency
    results = optimizer.measure_latency(
        queries=queries,
        methods=optimizations,
        n_runs=args.n_runs,
        batch_size=args.batch_size
    )
    
    # Print summary
    print_summary(results)
    
    # Save results
    save_results(results, args.output_file)

if __name__ == "__main__":
    main()