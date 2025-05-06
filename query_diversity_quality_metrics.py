import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from typing import Dict, List, Set, Tuple
import pandas as pd
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

class QueryEvaluator:
    """
    A class for evaluating the diversity and quality of query selection methods.
    """
    
    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the QueryEvaluator with a sentence embedding model.
        
        Args:
            embedding_model (str): Sentence transformer model for generating embeddings.
        """
        self.model = SentenceTransformer(embedding_model)
        
    def _get_embeddings(self, queries: List[str]) -> np.ndarray:
        """Generate embeddings for a list of queries."""
        return self.model.encode(queries, show_progress_bar=True)
    
    def average_pairwise_similarity(self, queries: List[str]) -> float:
        """
        Calculate the average pairwise similarity between queries.
        Lower values indicate more diverse query sets.
        
        Args:
            queries (List[str]): List of query strings to evaluate.
            
        Returns:
            float: Average pairwise similarity (between 0 and 1).
        """
        if len(queries) <= 1:
            return 0.0
            
        embeddings = self._get_embeddings(queries)
        similarity_matrix = cosine_similarity(embeddings)
        
        # Remove self-similarities (diagonal)
        np.fill_diagonal(similarity_matrix, 0)
        
        # Calculate average similarity
        n = len(queries)
        total_similarity = np.sum(similarity_matrix) / (n * (n - 1))
        
        return total_similarity
    
    def embedding_space_coverage(self, 
                               selected_queries: List[str], 
                               all_queries: List[str]) -> float:
        """
        Measure how well the selected queries cover the embedding space of all queries.
        Higher values indicate better coverage.
        
        Args:
            selected_queries (List[str]): List of selected query strings.
            all_queries (List[str]): List of all original query strings.
            
        Returns:
            float: Coverage score (higher is better).
        """
        if not selected_queries or not all_queries:
            return 0.0
            
        # Get embeddings for both sets
        selected_embeddings = self._get_embeddings(selected_queries)
        all_embeddings = self._get_embeddings(all_queries)
        
        # For each query in the original set, find the minimum distance to any selected query
        similarity_matrix = cosine_similarity(all_embeddings, selected_embeddings)
        
        # For each original query, get the maximum similarity to any selected query
        max_similarities = np.max(similarity_matrix, axis=1)
        
        # Coverage is the average of these maximum similarities
        coverage = np.mean(max_similarities)
        
        return coverage
    
    def icd_code_coverage(self, 
                        selected_queries: List[str], 
                        all_queries: List[str],
                        query_to_icd: Dict[str, List[str]]) -> float:
        """
        Calculate the coverage of ICD codes by the selected queries.
        Higher values indicate better medical topic coverage.
        
        Args:
            selected_queries (List[str]): List of selected query strings.
            all_queries (List[str]): List of all original query strings.
            query_to_icd (Dict[str, List[str]]): Dictionary mapping queries to ICD codes.
            
        Returns:
            float: ICD coverage ratio (0 to 1).
        """
        # Get all unique ICD codes in the original query set
        all_icd_codes = set()
        for query in all_queries:
            if query in query_to_icd:
                all_icd_codes.update(query_to_icd[query])
        
        # Get all unique ICD codes in the selected query set
        selected_icd_codes = set()
        for query in selected_queries:
            if query in query_to_icd:
                selected_icd_codes.update(query_to_icd[query])
        
        # If there are no ICD codes, return 0
        if not all_icd_codes:
            return 0.0
        
        # Calculate coverage ratio
        coverage_ratio = len(selected_icd_codes) / len(all_icd_codes)
        
        return coverage_ratio
    
    def silhouette_evaluation(self, queries: List[str]) -> float:
        """
        Evaluate the quality of query clustering using silhouette score.
        Higher values indicate more well-defined clusters.
        
        Args:
            queries (List[str]): List of query strings to evaluate.
            
        Returns:
            float: Silhouette score (-1 to 1, higher is better).
        """
        if len(queries) < 4:  # Need at least 4 points for meaningful clustering
            return 0.0
            
        # Get embeddings
        embeddings = self._get_embeddings(queries)
        
        # Use KMeans to cluster
        from sklearn.cluster import KMeans
        
        # Determine appropriate number of clusters
        n_clusters = min(int(np.sqrt(len(queries))), len(queries) - 1)
        n_clusters = max(2, n_clusters)  # At least 2 clusters
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Calculate silhouette score
        score = silhouette_score(embeddings, cluster_labels)
        
        return score
    
    def evaluate_methods(self, 
                        specialty_queries: Dict[str, List[str]],
                        similarity_results: Dict[str, List[str]], 
                        cluster_results: Dict[str, List[str]],
                        query_to_icd: Dict[str, List[str]] = None) -> pd.DataFrame:
        """
        Evaluate both query selection methods using multiple metrics.
        
        Args:
            specialty_queries (Dict[str, List[str]]): Original queries by specialty.
            similarity_results (Dict[str, List[str]]): Results from similarity-based method.
            cluster_results (Dict[str, List[str]]): Results from cluster-based method.
            query_to_icd (Dict[str, List[str]], optional): Mapping from queries to ICD codes.
            
        Returns:
            pd.DataFrame: DataFrame with evaluation metrics for both methods.
        """
        results = []
        
        for specialty in specialty_queries:
            all_queries = specialty_queries[specialty]
            sim_queries = similarity_results.get(specialty, [])
            clust_queries = cluster_results.get(specialty, [])
            
            # Skip if any result set is empty
            if not sim_queries or not clust_queries:
                continue
                
            # Calculate metrics for similarity-based method
            sim_avg_similarity = self.average_pairwise_similarity(sim_queries)
            sim_coverage = self.embedding_space_coverage(sim_queries, all_queries)
            sim_silhouette = self.silhouette_evaluation(sim_queries)
            
            # Calculate metrics for cluster-based method
            clust_avg_similarity = self.average_pairwise_similarity(clust_queries)
            clust_coverage = self.embedding_space_coverage(clust_queries, all_queries)
            clust_silhouette = self.silhouette_evaluation(clust_queries)
            
            # Calculate ICD coverage if mapping is provided
            sim_icd_coverage = 0.0
            clust_icd_coverage = 0.0
            if query_to_icd:
                sim_icd_coverage = self.icd_code_coverage(sim_queries, all_queries, query_to_icd)
                clust_icd_coverage = self.icd_code_coverage(clust_queries, all_queries, query_to_icd)
            
            # Add results to list
            results.append({
                'Specialty': specialty,
                'Method': 'Similarity-based',
                'Num_Queries': len(sim_queries),
                'Avg_Similarity': sim_avg_similarity,
                'Coverage': sim_coverage,
                'Silhouette': sim_silhouette,
                'ICD_Coverage': sim_icd_coverage
            })
            
            results.append({
                'Specialty': specialty,
                'Method': 'Cluster-based',
                'Num_Queries': len(clust_queries),
                'Avg_Similarity': clust_avg_similarity, 
                'Coverage': clust_coverage,
                'Silhouette': clust_silhouette,
                'ICD_Coverage': clust_icd_coverage
            })
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        return results_df
    
    def visualize_comparison(self, evaluation_df: pd.DataFrame) -> None:
        """
        Visualize the comparison between the two methods.
        
        Args:
            evaluation_df (pd.DataFrame): DataFrame with evaluation metrics.
        """
        # Prepare data for visualization
        metrics = ['Avg_Similarity', 'Coverage', 'Silhouette', 'ICD_Coverage']
        
        # Create a figure with subplots
        fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 12))
        
        for i, metric in enumerate(metrics):
            # Create a grouped bar plot for each metric
            sns.barplot(x='Specialty', y=metric, hue='Method', data=evaluation_df, ax=axes[i])
            axes[i].set_title(f'{metric} by Specialty and Method')
            axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')
            
            # For Avg_Similarity, lower is better, so note this on the plot
            if metric == 'Avg_Similarity':
                axes[i].text(0.5, 0.9, 'Lower is better', transform=axes[i].transAxes, 
                            ha='center', bbox=dict(facecolor='white', alpha=0.5))
            else:
                axes[i].text(0.5, 0.9, 'Higher is better', transform=axes[i].transAxes, 
                            ha='center', bbox=dict(facecolor='white', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('method_comparison.png')
        plt.close()

    def map_queries_to_icd(self, 
                          specialty_queries: Dict[str, List[str]], 
                          medical_nlp_model=None) -> Dict[str, List[str]]:
        """
        Map queries to ICD codes using a medical NLP model.
        This is a placeholder implementation - you'll need to integrate with a real medical NLP system.
        
        Args:
            specialty_queries (Dict[str, List[str]]): Dictionary mapping specialties to queries.
            medical_nlp_model: A model for mapping text to ICD codes (not implemented).
            
        Returns:
            Dict[str, List[str]]: Dictionary mapping queries to lists of ICD codes.
        """
        # This is a placeholder - in a real implementation, you would:
        # 1. Use a medical entity recognition model to identify conditions
        # 2. Map those conditions to ICD codes using a medical ontology
        # 3. Return the mapping from queries to ICD codes
        
        # For testing purposes, we'll create a mock mapping
        query_to_icd = {}
        
        # Mock ICD code mappings based on keywords
        icd_mapping = {
            'heart': ['I25', 'I21'],  # Ischemic heart disease, Acute myocardial infarction
            'chest pain': ['R07.9', 'I20'],  # Chest pain, unspecified; Angina pectoris
            'cardiac': ['I50', 'I46'],  # Heart failure, Cardiac arrest
            'coronary': ['I25.1'],  # Atherosclerotic heart disease
            'acne': ['L70'],  # Acne
            'eczema': ['L20'],  # Atopic dermatitis
            'rash': ['L30.9'],  # Dermatitis, unspecified
            'psoriasis': ['L40'],  # Psoriasis
            'mole': ['D22', 'C43'],  # Melanocytic nevi, Malignant melanoma
            'skin cancer': ['C44'],  # Other malignant neoplasms of skin
            'rosacea': ['L71']  # Rosacea
        }
        
        # For each specialty and query, assign mock ICD codes based on keywords
        for specialty, queries in specialty_queries.items():
            for query in queries:
                query_icd_codes = set()
                
                # Check for keywords in the query
                for keyword, codes in icd_mapping.items():
                    if keyword in query.lower():
                        query_icd_codes.update(codes)
                
                # If no codes found, assign a generic code for the specialty
                if not query_icd_codes:
                    if 'cardio' in specialty.lower():
                        query_icd_codes.add('I99')  # Other disorders of circulatory system
                    elif 'derma' in specialty.lower():
                        query_icd_codes.add('L99')  # Other disorders of skin
                    else:
                        query_icd_codes.add('R69')  # Unknown and unspecified causes of morbidity
                
                query_to_icd[query] = list(query_icd_codes)
        
        return query_to_icd


# Example usage:
if __name__ == "__main__":
    # First, import the query selector and run the selection methods
    from Query_Diversity_Selection_Algorithm import QueryDiversitySelector
    
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
    
    # Initialize the evaluator
    evaluator = QueryEvaluator()
    
    # Map queries to ICD codes (mock implementation)
    query_to_icd = evaluator.map_queries_to_icd(medical_specialties)
    
    # Evaluate the methods
    evaluation_df = evaluator.evaluate_methods(
        medical_specialties,
        similarity_results,
        cluster_results,
        query_to_icd
    )
    
    # Print the evaluation results
    print("\nEvaluation Results:")
    print(evaluation_df)
    
    # Visualize the comparison
    evaluator.visualize_comparison(evaluation_df)
    
    # Print a summary interpretation
    print("\nInterpretation of Results:")
    
    # Calculate average metrics for each method
    method_summary = evaluation_df.groupby('Method').mean()
    print("\nAverage metrics by method:")
    print(method_summary[['Avg_Similarity', 'Coverage', 'Silhouette', 'ICD_Coverage']])
    
    # Determine which method is better overall
    better_similarity = method_summary.loc['Cluster-based', 'Avg_Similarity'] < method_summary.loc['Similarity-based', 'Avg_Similarity']
    better_coverage = method_summary.loc['Cluster-based', 'Coverage'] > method_summary.loc['Similarity-based', 'Coverage']
    better_silhouette = method_summary.loc['Cluster-based', 'Silhouette'] > method_summary.loc['Similarity-based', 'Silhouette']
    better_icd = method_summary.loc['Cluster-based', 'ICD_Coverage'] > method_summary.loc['Similarity-based', 'ICD_Coverage']
    
    cluster_wins = sum([better_similarity, better_coverage, better_silhouette, better_icd])
    similarity_wins = 4 - cluster_wins
    
    print(f"\nCluster-based method wins in {cluster_wins} metrics")
    print(f"Similarity-based method wins in {similarity_wins} metrics")
    
    if cluster_wins > similarity_wins:
        print("\nConclusion: The cluster-based method appears to perform better overall.")
    elif similarity_wins > cluster_wins:
        print("\nConclusion: The similarity-based method appears to perform better overall.")
    else:
        print("\nConclusion: Both methods perform similarly overall, but they may excel in different aspects.")
