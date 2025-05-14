I'll help you find the optimal threshold to maximize your F1 score for your retrieval system. This is a common task in ML evaluation, especially for similarity-based retrieval systems.

Let me walk you through an approach to solve this:

1. Calculate similarity scores between your query embeddings and reference ICD code embeddings
2. For different thresholds, calculate precision, recall, and F1 scores
3. Find the threshold that maximizes the F1 score

Here's a Python implementation that should work for your use case:

```python
import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score
from sklearn.metrics.pairwise import cosine_similarity

def find_optimal_threshold(query_embeddings, reference_embeddings, ground_truth_icd_codes, reference_icd_codes):
    """
    Find the optimal threshold that maximizes F1 score for a retrieval system.
    
    Parameters:
    - query_embeddings: numpy array of shape (n_queries, embedding_dim)
    - reference_embeddings: numpy array of shape (n_references, embedding_dim)
    - ground_truth_icd_codes: list of lists, where each inner list contains ground truth ICD codes for a query
    - reference_icd_codes: list of ICD codes corresponding to reference_embeddings
    
    Returns:
    - optimal_threshold: float, the threshold that maximizes F1 score
    - max_f1: float, the maximum F1 score achieved
    """
    # Calculate similarity matrix between queries and references
    similarity_matrix = cosine_similarity(query_embeddings, reference_embeddings)
    
    # Convert ground truth to binary matrix
    n_queries = len(ground_truth_icd_codes)
    n_references = len(reference_icd_codes)
    y_true = np.zeros((n_queries, n_references), dtype=int)
    
    for i, gt_codes in enumerate(ground_truth_icd_codes):
        for gt_code in gt_codes:
            if gt_code in reference_icd_codes:
                j = reference_icd_codes.index(gt_code)
                y_true[i, j] = 1
    
    # Flatten the matrices for precision-recall curve
    y_true_flat = y_true.flatten()
    similarity_flat = similarity_matrix.flatten()
    
    # Calculate precision, recall, and thresholds
    precision, recall, thresholds = precision_recall_curve(y_true_flat, similarity_flat)
    
    # Calculate F1 scores for each threshold
    f1_scores = []
    for threshold in thresholds:
        y_pred_flat = (similarity_flat >= threshold).astype(int)
        f1 = f1_score(y_true_flat, y_pred_flat)
        f1_scores.append(f1)
    
    # Find the threshold that maximizes F1 score
    max_f1_idx = np.argmax(f1_scores)
    max_f1 = f1_scores[max_f1_idx]
    optimal_threshold = thresholds[max_f1_idx]
    
    return optimal_threshold, max_f1
```

You can then visualize the results to see how F1 score changes with different thresholds:

```python
import matplotlib.pyplot as plt

def plot_f1_vs_threshold(thresholds, f1_scores, optimal_threshold, max_f1):
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_scores)
    plt.axvline(x=optimal_threshold, color='r', linestyle='--', 
                label=f'Optimal threshold: {optimal_threshold:.3f}')
    plt.scatter([optimal_threshold], [max_f1], color='r', s=100)
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs. Threshold')
    plt.legend()
    plt.grid(True)
    plt.show()
```

To use these functions with your specific data:

```python
# Assuming you have:
# - embedding_model: your trained embedding model
# - ground_truth_icd_codes: list of lists of ground truth ICD codes
# - reference_icd_lookup: dictionary mapping ICD codes to embeddings

# Extract reference ICD codes and embeddings
reference_icd_codes = list(reference_icd_lookup.keys())
reference_embeddings = np.array(list(reference_icd_lookup.values()))

# Generate query embeddings (if not already available)
# query_embeddings = ...

# Find optimal threshold
optimal_threshold, max_f1 = find_optimal_threshold(
    query_embeddings, 
    reference_embeddings, 
    ground_truth_icd_codes, 
    reference_icd_codes
)

print(f"Optimal threshold: {optimal_threshold:.4f}")
print(f"Maximum F1 score: {max_f1:.4f}")

# Plot F1 vs threshold
plot_f1_vs_threshold(thresholds, f1_scores, optimal_threshold, max_f1)
```

You could also extend this approach to implement a cross-validation strategy to ensure your threshold generalizes well:

```python
from sklearn.model_selection import KFold

def cross_validate_threshold(query_embeddings, reference_embeddings, ground_truth_icd_codes, reference_icd_codes, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_thresholds = []
    
    indices = np.arange(len(query_embeddings))
    
    for train_idx, val_idx in kf.split(indices):
        train_query_emb = query_embeddings[train_idx]
        train_gt_codes = [ground_truth_icd_codes[i] for i in train_idx]
        
        threshold, _ = find_optimal_threshold(
            train_query_emb,
            reference_embeddings,
            train_gt_codes,
            reference_icd_codes
        )
        
        cv_thresholds.append(threshold)
    
    return np.mean(cv_thresholds), cv_thresholds
```

Would you like me to explain any part of this approach in more detail?​​​​​​​​​​​​​​​​
