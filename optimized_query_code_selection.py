import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import tqdm

def get_relevant_codes(selector, queries: List[str], icd_codes_df: pd.DataFrame, 
                     code_embeddings=None, top_k: int = 10) -> Dict[str, List[Tuple[str, str, float]]]:
    """
    Find the most relevant ICD codes for each query based on embedding similarity.
    
    Args:
        selector: An instance of QueryDiversitySelector that provides the _get_embeddings method
        queries: List of search queries
        icd_codes_df: DataFrame with ICD codes and descriptions (columns: 'code', 'description')
        code_embeddings: Pre-computed embeddings for ICD code descriptions (optional)
        top_k: Number of top relevant codes to return for each query
        
    Returns:
        Dictionary mapping each query to a list of tuples (icd_code, description, similarity_score)
    """
    # Validate input DataFrame has required columns
    if 'code' not in icd_codes_df.columns or 'description' not in icd_codes_df.columns:
        raise ValueError("icd_codes_df must have 'code' and 'description' columns")
    
    # Get query embeddings
    print(f"Generating embeddings for {len(queries)} queries...")
    query_embeddings = selector._get_embeddings(queries)
    
    # Get or use pre-computed ICD code description embeddings
    if code_embeddings is None:
        print(f"Generating embeddings for {len(icd_codes_df)} ICD code descriptions...")
        code_descriptions = icd_codes_df['description'].tolist()
        code_embeddings = selector._get_embeddings(code_descriptions)
    else:
        print("Using pre-computed ICD code embeddings...")
    
    # Calculate cosine similarity between each query and all ICD codes
    print("Calculating similarities and finding top matches...")
    query_to_codes = {}
    
    for i, query in enumerate(queries):
        # Calculate similarity between current query and all code descriptions
        similarities = cosine_similarity([query_embeddings[i]], code_embeddings)[0]
        
        # Get top k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Store results as (code, description, similarity) tuples
        relevant_codes = []
        for idx in top_indices:
            code = icd_codes_df.iloc[idx]['code']
            description = icd_codes_df.iloc[idx]['description']
            similarity = similarities[idx]
            relevant_codes.append((code, description, similarity))
        
        query_to_codes[query] = relevant_codes
    
    return query_to_codes

def map_query_to_icd(selector, specialty_queries: Dict[str, List[str]], icd_codes_df: pd.DataFrame, top_k: int = 10) -> Dict[str, Dict[str, List[Tuple[str, str, float]]]]:
    """
    Map queries to relevant ICD codes for each medical specialty.
    Optimized to generate ICD code embeddings only once.
    
    Args:
        selector: QueryDiversitySelector instance
        specialty_queries: Dictionary mapping specialties to lists of queries
        icd_codes_df: DataFrame with ICD codes and descriptions
        top_k: Number of top relevant codes to return for each query
        
    Returns:
        Nested dictionary: specialty -> {query -> [(code, description, similarity)]}
    """
    specialty_to_query_codes = {}
    
    # Generate ICD code description embeddings once
    print(f"Generating embeddings for {len(icd_codes_df)} ICD code descriptions (one-time operation)...")
    code_descriptions = icd_codes_df['description'].tolist()
    code_embeddings = selector._get_embeddings(code_descriptions)
    
    # Process each specialty with the pre-computed embeddings
    for specialty, queries in specialty_queries.items():
        print(f"\nProcessing specialty: {specialty} with {len(queries)} queries")
        
        # Get relevant codes for all queries in this specialty, reusing code embeddings
        query_to_codes = get_relevant_codes(
            selector, 
            queries, 
            icd_codes_df, 
            code_embeddings=code_embeddings,  # Pass pre-computed embeddings
            top_k=top_k
        )
        
        specialty_to_query_codes[specialty] = query_to_codes
    
    return specialty_to_query_codes

def format_icd_results(specialty_to_query_codes: Dict[str, Dict[str, List[Tuple[str, str, float]]]], max_queries_per_specialty: int = 5, max_codes_per_query: int = 3) -> None:
    """
    Print a formatted summary of the mapping results.
    
    Args:
        specialty_to_query_codes: Nested dictionary from map_query_to_icd
        max_queries_per_specialty: Maximum number of queries to show per specialty
        max_codes_per_query: Maximum number of codes to show per query
    """
    for specialty, query_codes in specialty_to_query_codes.items():
        print(f"\n=== {specialty} ===")
        
        # Show results for a limited number of queries
        for i, (query, codes) in enumerate(query_codes.items()):
            if i >= max_queries_per_specialty:
                break
                
            print(f"\nQuery: {query}")
            
            # Show top codes for this query
            for j, (code, description, similarity) in enumerate(codes):
                if j >= max_codes_per_query:
                    break
                    
                print(f"  - {code}: {description} (similarity: {similarity:.4f})")
        
        remaining_queries = len(query_codes) - max_queries_per_specialty
        if remaining_queries > 0:
            print(f"\n... and {remaining_queries} more queries")

def create_query_to_icd_mapping(specialty_query_dict: Dict[str, List[str]], icd_codes_df: pd.DataFrame, selector, evaluator=None) -> Dict[str, List[str]]:
    """
    Create a mapping from queries to ICD codes and optionally evaluate the mapping.
    
    Args:
        specialty_query_dict: Dictionary mapping specialties to lists of queries
        icd_codes_df: DataFrame with ICD codes and descriptions
        selector: QueryDiversitySelector instance
        evaluator: Optional QueryEvaluator instance
        
    Returns:
        Dictionary mapping each query to a list of ICD codes
    """
    # Map queries to ICD codes
    specialty_to_query_codes = map_query_to_icd(selector, specialty_query_dict, icd_codes_df)
    
    # Format for display
    format_icd_results(specialty_to_query_codes)
    
    # Convert to the format needed for evaluation
    query_to_icd = {}
    for specialty, query_codes in specialty_to_query_codes.items():
        for query, codes in query_codes.items():
            query_to_icd[query] = [code for code, _, _ in codes]
    
    return query_to_icd

# Example usage
if __name__ == "__main__":
    # Import required modules
    from Query_Diversity_Selection_Algorithm import QueryDiversitySelector
    
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
    
    # Initialize the selector
    embedding_model = 'all-MiniLM-L6-v2'  # Change to your preferred model
    selector = QueryDiversitySelector(embedding_model=embedding_model)
    
    # Create the query to ICD mapping
    query_to_icd = create_query_to_icd_mapping(specialty_query_dict, icd_codes_df, selector)
    
    # Run diversity selection algorithms
    similarity_results, cluster_results = selector.run_comparison(
        specialty_query_dict, 
        similarity_threshold=0.75, 
        k_clusters=3
    )
    
    # If evaluator is available, evaluate the methods
    try:
        from Query_Diversity_Evaluation import QueryEvaluator
        
        evaluator = QueryEvaluator(embedding_model=embedding_model)
        evaluation_df = evaluator.evaluate_methods(
            specialty_query_dict,
            similarity_results,
            cluster_results,
            query_to_icd
        )
        
        print("\nEvaluation Results:")
        print(evaluation_df)
        
        # Visualize the comparison
        evaluator.visualize_comparison(evaluation_df)
        
    except ImportError as e:
        print(f"\nEvaluator not available: {e}")
        print("Skipping evaluation phase.")
