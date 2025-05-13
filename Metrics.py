I see you're working on an evaluation function for a retrieval task involving ICD codes. This is a common challenge in medical NLP tasks where exact matches don't fully capture relevance. Let me help you implement a better function that accounts for code families.

Looking at your current implementation for precision and recall@k metrics, I can enhance it to handle ICD code families. Here's a function that will consider codes from the same family as relevant:

```python
def compute_metrics_at_k(relevant_items, retrieved_items, k):
    """
    Compute precision@k and recall@k for ICD code retrieval
    
    Args:
        relevant_items (list): List of relevant ICD codes (ground truth)
        retrieved_items (list): List of retrieved/predicted ICD codes
        k (int): Number of top predictions to consider
    
    Returns:
        dict: Dictionary containing precision@k and recall@k values
    """
    # Ensure we only consider the top k retrieved items
    retrieved_at_k = retrieved_items[:k]
    
    if len(retrieved_at_k) == 0:
        return {"precision@k": 0.0, "recall@k": 0.0}
    
    if len(relevant_items) == 0:
        return {"precision@k": 0.0, "recall@k": 0.0}
    
    # Get code families for comparison
    relevant_families = set(get_code_family(code) for code in relevant_items)
    retrieved_families = set(get_code_family(code) for code in retrieved_at_k)
    
    # For exact matching individual codes
    exact_matches = set(relevant_items).intersection(set(retrieved_at_k))
    exact_match_count = len(exact_matches)
    
    # For family matching - codes that share the same family
    family_matches = relevant_families.intersection(retrieved_families)
    family_match_count = len(family_matches)
    
    # Calculate metrics with family matching
    precision = family_match_count / len(retrieved_at_k)
    recall = family_match_count / len(relevant_families)
    
    # You could also create a weighted measure that values exact matches more
    # precision_weighted = (exact_match_count + 0.5 * (family_match_count - exact_match_count)) / len(retrieved_at_k)
    
    return {
        "precision@k": precision,
        "recall@k": recall,
        "exact_match_count": exact_match_count,
        "family_match_count": family_match_count
    }

def get_code_family(icd_code):
    """
    Extract the family portion of an ICD code
    For example: 'F32.44' -> 'F32'
    
    Args:
        icd_code (str): An ICD code
    
    Returns:
        str: The family portion of the code
    """
    # Split by dot and take the first part
    if '.' in icd_code:
        return icd_code.split('.')[0]
    
    # For codes without dots, take the letter prefix and first digits
    # This might need adjustment based on your specific ICD version
    import re
    match = re.match(r'([A-Z]\d+)', icd_code)
    if match:
        return match.group(1)
    
    # Return as is if pattern doesn't match
    return icd_code
```

This implementation:

1. Extracts code families from both relevant and retrieved items
2. Calculates matches both at the exact code level and at the family level
3. Provides metrics based on family matches, which addresses your requirement to consider codes like F32.44 and F32.41 as relevant to each other
4. Returns a dictionary with both precision and recall metrics

You could further refine this by:

1. Adding a hierarchical similarity measure that considers the depth of matching in the ICD hierarchy
2. Implementing a weighted scoring system where closer matches get higher scores
3. Adding a configurable similarity threshold for family matching

Would you like me to explain any specific part in more detail or adjust the implementation to better fit your specific use case?​​​​​​​​​​​​​​​​
