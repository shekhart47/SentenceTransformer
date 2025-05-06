import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set, Union
from tqdm import tqdm
import torch
from sklearn.metrics.pairwise import cosine_similarity
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer

class DistributedQueryEmbeddingMatcher:
    """
    A class for efficient, distributed generation of embeddings and matching queries to ICD codes
    using multiple GPUs.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', batch_size: int = 128):
        """
        Initialize the distributed embedding and matching system.
        
        Args:
            model_name: Name of the SentenceTransformer model to use
            batch_size: Batch size for embedding generation
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.device_count = torch.cuda.device_count()
        self.models = {}
        self.code_embeddings_cache = {}
        
        # Print GPU info for debugging
        print(f"Found {self.device_count} CUDA devices")
        for i in range(self.device_count):
            print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
            
        # Pre-load models on all available devices
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize model instances on each GPU."""
        print("Initializing models on all GPUs...")
        for device_id in range(self.device_count):
            device = f"cuda:{device_id}"
            self.models[device_id] = SentenceTransformer(self.model_name)
            self.models[device_id].to(device)
        print("All models initialized")
            
    def _split_data_for_gpus(self, data: List[str]) -> List[List[str]]:
        """Split data into chunks for processing on multiple GPUs."""
        chunk_size = len(data) // self.device_count
        if chunk_size == 0:
            # If data is smaller than number of GPUs, assign all to first GPU
            return [data] + [[] for _ in range(self.device_count - 1)]
            
        chunks = []
        for i in range(self.device_count - 1):
            chunks.append(data[i * chunk_size:(i + 1) * chunk_size])
        # Last chunk gets the remainder
        chunks.append(data[(self.device_count - 1) * chunk_size:])
        return chunks
        
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts using all available GPUs.
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            NumPy array of embeddings
        """
        if not texts:
            return np.array([])
            
        # Split data across GPUs
        data_chunks = self._split_data_for_gpus(texts)
        
        # Store results
        all_embeddings = []
        
        # Use ThreadPoolExecutor for concurrent processing
        with ThreadPoolExecutor(max_workers=self.device_count) as executor:
            # Submit tasks for each GPU
            futures = []
            for device_id, chunk in enumerate(data_chunks):
                if not chunk:  # Skip empty chunks
                    continue
                futures.append(
                    executor.submit(
                        self._generate_embeddings_on_device, 
                        device_id, 
                        chunk
                    )
                )
            
            # Collect results as they complete
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing GPU chunks"):
                embeddings = future.result()
                all_embeddings.append(embeddings)
        
        # Concatenate all embeddings
        if not all_embeddings:
            return np.array([])
        return np.vstack(all_embeddings)
    
    def _generate_embeddings_on_device(self, device_id: int, texts: List[str]) -> np.ndarray:
        """Generate embeddings on a specific GPU device."""
        if not texts:
            return np.array([])
            
        device = f"cuda:{device_id}"
        model = self.models[device_id]
        
        # Process in batches to avoid OOM errors
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            with torch.no_grad():
                embeddings = model.encode(
                    batch,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    batch_size=self.batch_size
                )
            all_embeddings.append(embeddings)
            
        # Clear CUDA cache to prevent memory leaks
        torch.cuda.empty_cache()
        
        return np.vstack(all_embeddings) if all_embeddings else np.array([])
    
    def generate_code_description_embeddings(self, code_descriptions: List[str]) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for ICD code descriptions and cache them.
        
        Args:
            code_descriptions: List of ICD code descriptions
            
        Returns:
            Dictionary mapping descriptions to their embeddings
        """
        start_time = time.time()
        print(f"Generating embeddings for {len(code_descriptions)} code descriptions...")
        
        # Generate embeddings using all GPUs
        embeddings = self.generate_embeddings(code_descriptions)
        
        # Create description to embedding mapping
        description_to_embedding = {}
        for desc, embedding in zip(code_descriptions, embeddings):
            description_to_embedding[desc] = embedding
            
        elapsed = time.time() - start_time
        print(f"Generated embeddings in {elapsed:.2f} seconds")
        
        return description_to_embedding
    
    def find_relevant_codes(self, 
                          queries: List[str], 
                          icd_codes_df: pd.DataFrame,
                          description_embeddings: Dict[str, np.ndarray] = None,
                          top_k: int = 10) -> Dict[str, List[Tuple[str, str, float]]]:
        """
        Find the most relevant ICD codes for each query.
        
        Args:
            queries: List of search queries
            icd_codes_df: DataFrame with ICD codes and descriptions
            description_embeddings: Pre-computed embeddings for ICD descriptions (optional)
            top_k: Number of top codes to return for each query
            
        Returns:
            Dictionary mapping queries to lists of (code, description, similarity) tuples
        """
        # Generate or use cached description embeddings
        if description_embeddings is None:
            descriptions = icd_codes_df['description'].tolist()
            description_embeddings = self.generate_code_description_embeddings(descriptions)
        
        # Generate query embeddings
        start_time = time.time()
        print(f"Generating embeddings for {len(queries)} queries...")
        query_embeddings = self.generate_embeddings(queries)
        print(f"Query embeddings generated in {time.time() - start_time:.2f} seconds")
        
        # Convert description embeddings to matrix for batch similarity
        descriptions = list(description_embeddings.keys())
        description_embedding_matrix = np.vstack([description_embeddings[desc] for desc in descriptions])
        
        # Calculate similarities and find top matches
        start_time = time.time()
        print("Calculating similarities and finding top matches...")
        
        # Create a mapping from descriptions to their indices and codes
        desc_to_idx = {desc: i for i, desc in enumerate(descriptions)}
        desc_to_code = {}
        for _, row in icd_codes_df.iterrows():
            desc_to_code[row['description']] = row['code']
        
        # Use batch similarity calculation for efficiency
        results = {}
        for i, query in enumerate(queries):
            similarities = cosine_similarity([query_embeddings[i]], description_embedding_matrix)[0]
            
            # Get top k indices
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            # Store results as (code, description, similarity) tuples
            query_results = []
            for idx in top_indices:
                description = descriptions[idx]
                code = desc_to_code.get(description, "Unknown")
                similarity = similarities[idx]
                query_results.append((code, description, similarity))
                
            results[query] = query_results
            
        print(f"Matching completed in {time.time() - start_time:.2f} seconds")
        return results
    
    def batch_process_specialties(self, 
                               specialty_queries: Dict[str, List[str]], 
                               icd_codes_df: pd.DataFrame,
                               top_k: int = 10) -> Dict[str, Dict[str, List[Tuple[str, str, float]]]]:
        """
        Process all specialties efficiently, with pre-computed ICD embeddings.
        
        Args:
            specialty_queries: Dictionary mapping specialties to lists of queries
            icd_codes_df: DataFrame with ICD codes and descriptions
            top_k: Number of top codes to return for each query
            
        Returns:
            Nested dictionary mapping specialties to query results
        """
        # Generate ICD description embeddings once
        descriptions = icd_codes_df['description'].tolist()
        description_embeddings = self.generate_code_description_embeddings(descriptions)
        
        # Process each specialty
        specialty_results = {}
        for specialty, queries in specialty_queries.items():
            print(f"\nProcessing specialty: {specialty} with {len(queries)} queries")
            
            # Find relevant codes for all queries in this specialty
            query_results = self.find_relevant_codes(
                queries, 
                icd_codes_df,
                description_embeddings=description_embeddings,
                top_k=top_k
            )
            
            specialty_results[specialty] = query_results
            
        return specialty_results

# Helper function to demonstrate usage and benchmark performance
def benchmark_distributed_embedding(specialty_queries, icd_codes_df, model_name='all-MiniLM-L6-v2', batch_size=128):
    """
    Benchmark the distributed embedding and matching process.
    
    Args:
        specialty_queries: Dictionary mapping specialties to lists of queries
        icd_codes_df: DataFrame with ICD codes and descriptions
        model_name: Name of the SentenceTransformer model
        batch_size: Batch size for embedding generation
    """
    print("\n=== PERFORMANCE BENCHMARK ===")
    
    # Count total queries
    total_queries = sum(len(queries) for queries in specialty_queries.values())
    print(f"Total specialties: {len(specialty_queries)}")
    print(f"Total queries: {total_queries}")
    print(f"Total ICD codes: {len(icd_codes_df)}")
    
    # Initialize the distributed matcher
    matcher = DistributedQueryEmbeddingMatcher(model_name=model_name, batch_size=batch_size)
    
    # Measure total processing time
    start_time = time.time()
    
    # Process all specialties
    results = matcher.batch_process_specialties(specialty_queries, icd_codes_df)
    
    # Calculate total time
    total_time = time.time() - start_time
    
    print("\n=== BENCHMARK RESULTS ===")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average time per query: {total_time / total_queries:.4f} seconds")
    print(f"Processing rate: {total_queries / total_time:.2f} queries/second")
    
    return results

# Example usage
if __name__ == "__main__":
    # Sample data
    specialty_query_dict = {
        "Cardiology": [
            "heart attack symptoms",
            "chest pain causes",
            "cardiac arrest vs heart attack",
            "heart disease prevention",
            "coronary artery blockage"
        ],
        "Dermatology": [
            "acne treatment",
            "eczema remedies",
            "skin rash identification",
            "psoriasis symptoms",
            "mole cancer signs"
        ]
    }
    
    # Sample ICD codes DataFrame
    data = {
        'code': ['I21', 'I20', 'I25', 'I50', 'I10', 'L70', 'L20', 'L30', 'L40', 'C43'],
        'description': [
            'Acute myocardial infarction',
            'Angina pectoris',
            'Chronic ischemic heart disease',
            'Heart failure',
            'Essential (primary) hypertension',
            'Acne',
            'Atopic dermatitis',
            'Other and unspecified dermatitis',
            'Psoriasis',
            'Malignant melanoma of skin'
        ]
    }
    icd_codes_df = pd.DataFrame(data)
    
    # Run benchmark
    results = benchmark_distributed_embedding(
        specialty_query_dict, 
        icd_codes_df,
        model_name='all-MiniLM-L6-v2',
        batch_size=128
    )
    
    # Print sample results
    print("\n=== SAMPLE RESULTS ===")
    for specialty, query_results in results.items():
        print(f"\n{specialty}:")
        for query, codes in list(query_results.items())[:2]:  # Show first 2 queries
            print(f"  Query: {query}")
            for code, desc, sim in codes[:3]:  # Show top 3 codes
                print(f"    - {code}: {desc} (similarity: {sim:.4f})")
