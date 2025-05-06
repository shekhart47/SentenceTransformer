import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Set, Tuple
import tqdm
import logging

class QueryDiversitySelector:
    """
    A class that implements algorithms for selecting diverse queries from a set
    of search queries grouped by medical specialties.
    """
    
    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the QueryDiversitySelector with a sentence embedding model.
        
        Args:
            embedding_model (str): The name of the sentence transformer model to use.
                                  Default is 'all-MiniLM-L6-v2' which is a good 
                                  balance of speed and quality.
        """
        self.model = SentenceTransformer(embedding_model)
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
    def _get_embeddings(self, queries: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of search queries.
        
        Args:
            queries (List[str]): List of search query strings.
            
        Returns:
            np.ndarray: Matrix of query embeddings with shape (n_queries, embedding_dim).
        """
        # Process queries in batches to avoid memory issues with large query sets
        self.logger.info(f"Generating embeddings for {len(queries)} queries...")
        return self.model.encode(queries, show_progress_bar=True)
    
    def similarity_based_selection(self, 
                                 specialty_queries: Dict[str, List[str]], 
                                 similarity_threshold: float = 0.8,
                                 max_queries_per_specialty: int = None) -> Dict[str, List[str]]:
        """
        Select diverse queries using a cosine similarity threshold approach.
        
        This algorithm:
        1. Computes embeddings for all queries in each specialty
        2. Builds a similarity matrix for each specialty
        3. Incrementally selects queries that are not too similar to already selected ones
        
        Args:
            specialty_queries (Dict[str, List[str]]): Dictionary mapping medical specialties 
                                                     to lists of search queries.
            similarity_threshold (float): Threshold above which queries are considered similar.
                                         Higher values allow more similar queries (less strict).
            max_queries_per_specialty (int, optional): Maximum number of queries to select per specialty.
                                                      If None, will select as many as meet the threshold.
                                                      
        Returns:
            Dict[str, List[str]]: Dictionary mapping specialties to filtered, diverse queries.
        """
        diverse_queries = {}
        
        for specialty, queries in specialty_queries.items():
            self.logger.info(f"Processing specialty: {specialty} with {len(queries)} queries")
            
            if len(queries) <= 1:
                diverse_queries[specialty] = queries
                continue
                
            # Get embeddings for all queries in this specialty
            embeddings = self._get_embeddings(queries)
            
            # Compute cosine similarity matrix
            similarity_matrix = cosine_similarity(embeddings)
            
            # Initialize data structures
            selected_indices = []
            remaining_indices = set(range(len(queries)))
            
            # Start with the query that has the lowest average similarity to others
            # This tends to select more "unique" queries to start with
            avg_similarities = np.mean(similarity_matrix, axis=1)
            first_idx = np.argmin(avg_similarities)
            selected_indices.append(first_idx)
            remaining_indices.remove(first_idx)
            
            # Iteratively select queries
            while remaining_indices and (max_queries_per_specialty is None or 
                                        len(selected_indices) < max_queries_per_specialty):
                # For each remaining query, check similarity with already selected ones
                too_similar = set()
                for idx in remaining_indices:
                    # If any similarity exceeds the threshold, mark for removal
                    for selected_idx in selected_indices:
                        if similarity_matrix[idx, selected_idx] > similarity_threshold:
                            too_similar.add(idx)
                            break
                
                # Remove all too similar queries from consideration
                remaining_indices -= too_similar
                
                # If no remaining queries, break
                if not remaining_indices:
                    break
                    
                # Select the query that has the lowest average similarity to already selected ones
                next_idx = -1
                min_avg_sim = float('inf')
                for idx in remaining_indices:
                    avg_sim = np.mean([similarity_matrix[idx, selected_idx] for selected_idx in selected_indices])
                    if avg_sim < min_avg_sim:
                        min_avg_sim = avg_sim
                        next_idx = idx
                        
                selected_indices.append(next_idx)
                remaining_indices.remove(next_idx)
            
            # Map indices back to queries
            diverse_queries[specialty] = [queries[idx] for idx in selected_indices]
            self.logger.info(f"Selected {len(diverse_queries[specialty])} diverse queries for {specialty}")
        
        return diverse_queries
    
    def cluster_based_selection(self, 
                             specialty_queries: Dict[str, List[str]], 
                             k_clusters: int = None,
                             queries_per_cluster: int = 1) -> Dict[str, List[str]]:
        """
        Select diverse queries using a clustering-based approach.
        
        This algorithm:
        1. Computes embeddings for all queries in each specialty
        2. Clusters the embeddings using KMeans
        3. Selects representative queries from each cluster (closest to centroids)
        
        Args:
            specialty_queries (Dict[str, List[str]]): Dictionary mapping medical specialties 
                                                     to lists of search queries.
            k_clusters (int, optional): Number of clusters to use for each specialty.
                                       If None, it will be dynamically set to sqrt(n) where n is 
                                       the number of queries in the specialty.
            queries_per_cluster (int): Number of queries to select from each cluster.
                                      
        Returns:
            Dict[str, List[str]]: Dictionary mapping specialties to filtered, diverse queries.
        """
        diverse_queries = {}
        
        for specialty, queries in specialty_queries.items():
            self.logger.info(f"Processing specialty: {specialty} with {len(queries)} queries")
            
            # Handle edge cases
            if len(queries) <= 1:
                diverse_queries[specialty] = queries
                continue
                
            # Get embeddings for all queries in this specialty
            embeddings = self._get_embeddings(queries)
            
            # Determine number of clusters if not specified
            num_clusters = k_clusters
            if num_clusters is None:
                # A common heuristic is to use the square root of the number of data points
                num_clusters = min(int(np.sqrt(len(queries))), len(queries) - 1)
                num_clusters = max(2, num_clusters)  # At least 2 clusters
            
            # Ensure we don't have more clusters than queries
            num_clusters = min(num_clusters, len(queries))
            
            # Apply KMeans clustering
            self.logger.info(f"Clustering {len(queries)} queries into {num_clusters} clusters")
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            # Get cluster centroids
            centroids = kmeans.cluster_centers_
            
            # For each cluster, find the queries closest to the centroid
            selected_queries = []
            for cluster_idx in range(num_clusters):
                # Get indices of queries in this cluster
                cluster_query_indices = np.where(cluster_labels == cluster_idx)[0]
                
                if len(cluster_query_indices) == 0:
                    continue
                
                # Get embeddings of queries in this cluster
                cluster_embeddings = embeddings[cluster_query_indices]
                
                # Calculate distance to centroid for each query in the cluster
                centroid = centroids[cluster_idx].reshape(1, -1)
                distances = cosine_similarity(cluster_embeddings, centroid).flatten()
                
                # Sort by similarity (higher is closer to centroid)
                sorted_indices = np.argsort(-distances)
                
                # Take the top N queries from this cluster
                top_n = min(queries_per_cluster, len(cluster_query_indices))
                for i in range(top_n):
                    original_idx = cluster_query_indices[sorted_indices[i]]
                    selected_queries.append(queries[original_idx])
            
            diverse_queries[specialty] = selected_queries
            self.logger.info(f"Selected {len(diverse_queries[specialty])} diverse queries for {specialty}")
            
        return diverse_queries

    def run_comparison(self, 
                     specialty_queries: Dict[str, List[str]], 
                     similarity_threshold: float = 0.8,
                     k_clusters: int = None) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """
        Run both algorithms and return their results for comparison.
        
        Args:
            specialty_queries (Dict[str, List[str]]): Dictionary mapping medical specialties 
                                                     to lists of search queries.
            similarity_threshold (float): Threshold for similarity-based selection.
            k_clusters (int, optional): Number of clusters for cluster-based selection.
            
        Returns:
            Tuple containing:
            - Dict[str, List[str]]: Results from similarity-based selection
            - Dict[str, List[str]]: Results from cluster-based selection
        """
        sim_results = self.similarity_based_selection(specialty_queries, similarity_threshold)
        cluster_results = self.cluster_based_selection(specialty_queries, k_clusters)
        
        return sim_results, cluster_results


# Example usage:
if __name__ == "__main__":
    # Example data
    medical_specialties = {
        "Cardiology": [
            "heart attack symptoms",
            "chest pain causes",
            "cardiac arrest vs heart attack",
            "heart disease prevention",
            "coronary artery blockage",
            "heart attack warning signs",
            "chest pain when breathing",
            "how to prevent heart failure",
            "symptoms of heart disease"
        ],
        "Dermatology": [
            "acne treatment",
            "eczema remedies",
            "skin rash identification",
            "psoriasis symptoms",
            "mole cancer signs",
            "how to treat acne",
            "best acne medications",
            "rosacea vs acne",
            "skin cancer screening"
        ]
    }
    
    # Initialize the selector
    selector = QueryDiversitySelector()
    
    # Run both algorithms
    similarity_results, cluster_results = selector.run_comparison(
        medical_specialties, 
        similarity_threshold=0.75, 
        k_clusters=3
    )
    
    # Print results
    print("\nSimilarity-based Selection Results:")
    for specialty, queries in similarity_results.items():
        print(f"\n{specialty}:")
        for q in queries:
            print(f"  - {q}")
    
    print("\nCluster-based Selection Results:")
    for specialty, queries in cluster_results.items():
        print(f"\n{specialty}:")
        for q in queries:
            print(f"  - {q}")
