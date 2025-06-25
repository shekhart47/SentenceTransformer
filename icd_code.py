import numpy as np
import faiss
from typing import Dict, List, Tuple
import pickle
import json

class ICDRetriever:
“””
Efficient ICD code retrieval using FAISS for cosine similarity search.
“””

```
def __init__(self, code_embeddings: Dict[str, np.ndarray]):
    """
    Initialize the retriever with ICD code embeddings.
    
    Args:
        code_embeddings: Dict with ICD codes as keys and embeddings as values
    """
    self.code_embeddings = code_embeddings
    self.icd_codes = list(code_embeddings.keys())
    self.embedding_dim = None
    self.index = None
    self._build_index()

def _build_index(self):
    """Build FAISS index for efficient similarity search."""
    print("Building FAISS index...")
    
    # Convert embeddings to numpy array
    embeddings_list = [self.code_embeddings[code] for code in self.icd_codes]
    embeddings_array = np.array(embeddings_list, dtype=np.float32)
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings_array)
    
    self.embedding_dim = embeddings_array.shape[1]
    
    # Create FAISS index (using Inner Product for normalized vectors = cosine similarity)
    self.index = faiss.IndexFlatIP(self.embedding_dim)
    self.index.add(embeddings_array)
    
    print(f"Index built with {len(self.icd_codes)} ICD codes, embedding dimension: {self.embedding_dim}")

def search_top_k(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
    """
    Find top-k most similar ICD codes for a query embedding.
    
    Args:
        query_embedding: Query embedding vector
        k: Number of top results to return
        
    Returns:
        List of tuples (icd_code, similarity_score)
    """
    # Ensure query embedding is the right shape and type
    query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
    
    # Normalize query embedding
    faiss.normalize_L2(query_embedding)
    
    # Search
    similarities, indices = self.index.search(query_embedding, k)
    
    # Convert results to list of tuples
    results = []
    for sim, idx in zip(similarities[0], indices[0]):
        if idx != -1:  # Valid index
            results.append((self.icd_codes[idx], float(sim)))
    
    return results

def batch_search(self, query_embeddings: Dict[str, np.ndarray], k: int = 10) -> Dict[str, List[Tuple[str, float]]]:
    """
    Batch search for multiple queries.
    
    Args:
        query_embeddings: Dict with query descriptions as keys and embeddings as values
        k: Number of top results per query
        
    Returns:
        Dict with query descriptions as keys and list of (icd_code, similarity) tuples as values
    """
    print(f"Processing {len(query_embeddings)} queries...")
    
    results = {}
    for i, (query_desc, query_emb) in enumerate(query_embeddings.items()):
        results[query_desc] = self.search_top_k(query_emb, k)
        
        # Progress indicator
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(query_embeddings)} queries")
    
    return results

def save_index(self, filepath: str):
    """Save the FAISS index to disk."""
    faiss.write_index(self.index, filepath)
    
    # Save metadata
    metadata = {
        'icd_codes': self.icd_codes,
        'embedding_dim': self.embedding_dim
    }
    with open(filepath + '.metadata', 'w') as f:
        json.dump(metadata, f)

def load_index(self, filepath: str):
    """Load FAISS index from disk."""
    self.index = faiss.read_index(filepath)
    
    # Load metadata
    with open(filepath + '.metadata', 'r') as f:
        metadata = json.load(f)
    
    self.icd_codes = metadata['icd_codes']
    self.embedding_dim = metadata['embedding_dim']
```

def main():
“””
Main function to demonstrate usage.
Replace with your actual data loading logic.
“””

```
# Example: Load your dictionaries
# text_sentence_embedding_dictionary = load_query_embeddings()  # Your function
# code_embeddings = load_icd_embeddings()  # Your function

# For demonstration, creating dummy data
print("Creating dummy data for demonstration...")

# Dummy ICD codes and embeddings
embedding_dim = 384  # Common dimension for sentence transformers
n_codes = 1000
code_embeddings = {
    f"ICD_{i:04d}": np.random.randn(embedding_dim).astype(np.float32)
    for i in range(n_codes)
}

# Dummy queries and embeddings
n_queries = 50
text_sentence_embedding_dictionary = {
    f"Query_{i}: Some medical condition description": np.random.randn(embedding_dim).astype(np.float32)
    for i in range(n_queries)
}

# Initialize retriever
retriever = ICDRetriever(code_embeddings)

# Perform batch search
results = retriever.batch_search(text_sentence_embedding_dictionary, k=10)

# Display results
print("\nTop 3 queries with their top 5 ICD matches:")
for i, (query, matches) in enumerate(list(results.items())[:3]):
    print(f"\nQuery: {query}")
    print("Top 5 ICD matches:")
    for j, (icd_code, similarity) in enumerate(matches[:5]):
        print(f"  {j+1}. {icd_code}: {similarity:.4f}")

# Save results
print("\nSaving results...")
with open('icd_retrieval_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Optional: Save index for future use
retriever.save_index('icd_faiss_index.bin')
print("FAISS index saved to 'icd_faiss_index.bin'")
```

def load_your_data():
“””
Template function to load your actual data.
Replace this with your data loading logic.
“””
# Example loading patterns:

```
# Option 1: If saved as pickle files
# with open('text_sentence_embedding_dictionary.pkl', 'rb') as f:
#     text_sentence_embedding_dictionary = pickle.load(f)
# with open('code_embeddings.pkl', 'rb') as f:
#     code_embeddings = pickle.load(f)

# Option 2: If saved as numpy files
# text_sentence_embedding_dictionary = np.load('query_embeddings.npz', allow_pickle=True)
# code_embeddings = np.load('icd_embeddings.npz', allow_pickle=True)

# Option 3: If embeddings are stored in a database or API
# text_sentence_embedding_dictionary = fetch_from_database('query_embeddings')
# code_embeddings = fetch_from_database('icd_embeddings')

pass
```

if **name** == “**main**”:
# For actual usage, replace the dummy data creation in main() with:
# text_sentence_embedding_dictionary, code_embeddings = load_your_data()

```
main()
```
