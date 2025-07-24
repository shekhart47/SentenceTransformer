import pandas as pd
import numpy as np
import json

def load_paraphrased_queries():
    """Load paraphrased queries from JSON file"""
    file_path = './specialty_paraphrased_queries_dict.json'
    with open(file_path, 'r') as file:
        paraphrased_production_queries = json.load(file)
    return paraphrased_production_queries

def get_summary_stats(specialty_paraphrased_queries_dict, use_specialty_query_dict):
    """Generate summary statistics (placeholder for your existing function)"""
    # This should return your DataFrame with the statistics
    # For now, I'll assume it returns the DataFrame you showed
    pass

def sample_queries_per_specialty(df, target_queries_per_specialty=250, paraphrased_threshold=50):
    """
    Sample queries with preference for paraphrased queries based on specified rules.
    
    Rules:
    1. Target: 250 queries per specialty
    2. If paraphrased <= 50: take all paraphrased, fill remaining with synthetic
    3. If total available < 250: use all queries (no sampling)
    4. If paraphrased = 0: sample only from synthetic
    5. If paraphrased > 200: give equal weight to both types
    6. Otherwise: prioritize paraphrased, fill remaining with synthetic
    
    Args:
        df: DataFrame with columns ['Specialties', 'Total_Queries_Paraphrased', 'Total_Queries_Synthetic', 'Total_Queries']
        target_queries_per_specialty: Target number of queries per specialty (default: 250)
        paraphrased_threshold: Threshold for taking all paraphrased queries (default: 50)
    
    Returns:
        DataFrame with sampling results
    """
    results = []
    
    for _, row in df.iterrows():
        specialty = row['Specialties']
        paraphrased_count = int(row['Total_Queries_Paraphrased'])
        synthetic_count = int(row['Total_Queries_Synthetic'])
        total_available = int(row['Total_Queries'])
        
        # Initialize sampling counts
        sampled_paraphrased = 0
        sampled_synthetic = 0
        sampling_strategy = ""
        
        # Rule 3: If total available < target, use all queries
        if total_available < target_queries_per_specialty:
            sampled_paraphrased = paraphrased_count
            sampled_synthetic = synthetic_count
            sampling_strategy = "Use all available (insufficient total)"
            
        # Rule 4: If no paraphrased queries, sample only from synthetic
        elif paraphrased_count == 0:
            sampled_synthetic = min(target_queries_per_specialty, synthetic_count)
            sampling_strategy = "Synthetic only (no paraphrased available)"
            
        # Rule 2: If paraphrased <= threshold, take all paraphrased
        elif paraphrased_count <= paraphrased_threshold:
            sampled_paraphrased = paraphrased_count
            remaining_needed = target_queries_per_specialty - sampled_paraphrased
            sampled_synthetic = min(remaining_needed, synthetic_count)
            sampling_strategy = f"All paraphrased + synthetic (paraphrased <= {paraphrased_threshold})"
            
        # Rule 5: If paraphrased > 200, give equal weight
        elif paraphrased_count > 200:
            target_each = target_queries_per_specialty // 2  # 125 each
            sampled_paraphrased = min(target_each, paraphrased_count)
            sampled_synthetic = min(target_each, synthetic_count)
            
            # If one type has fewer than target_each, allocate remaining to the other
            total_so_far = sampled_paraphrased + sampled_synthetic
            if total_so_far < target_queries_per_specialty:
                remaining = target_queries_per_specialty - total_so_far
                if sampled_paraphrased < target_each:
                    # Synthetic was limited, try to get more paraphrased
                    additional_paraphrased = min(remaining, paraphrased_count - sampled_paraphrased)
                    sampled_paraphrased += additional_paraphrased
                else:
                    # Paraphrased was limited, try to get more synthetic
                    additional_synthetic = min(remaining, synthetic_count - sampled_synthetic)
                    sampled_synthetic += additional_synthetic
            
            sampling_strategy = "Equal weight (paraphrased > 200)"
            
        # Rule 6: Default case - prioritize paraphrased, fill with synthetic
        else:
            # Prioritize paraphrased - aim for about 70% if possible
            preferred_paraphrased = min(
                int(target_queries_per_specialty * 0.7),
                paraphrased_count
            )
            sampled_paraphrased = preferred_paraphrased
            remaining_needed = target_queries_per_specialty - sampled_paraphrased
            sampled_synthetic = min(remaining_needed, synthetic_count)
            
            # If we still need more and have more paraphrased available
            total_so_far = sampled_paraphrased + sampled_synthetic
            if total_so_far < target_queries_per_specialty and paraphrased_count > sampled_paraphrased:
                additional_needed = target_queries_per_specialty - total_so_far
                additional_paraphrased = min(additional_needed, paraphrased_count - sampled_paraphrased)
                sampled_paraphrased += additional_paraphrased
            
            sampling_strategy = "Prioritize paraphrased, fill with synthetic"
        
        total_sampled = sampled_paraphrased + sampled_synthetic
        paraphrased_ratio = sampled_paraphrased / total_sampled if total_sampled > 0 else 0
        
        results.append({
            'Specialties': specialty,
            'Original_Paraphrased': paraphrased_count,
            'Original_Synthetic': synthetic_count,
            'Original_Total': total_available,
            'Sampled_Paraphrased': sampled_paraphrased,
            'Sampled_Synthetic': sampled_synthetic,
            'Total_Sampled': total_sampled,
            'Paraphrased_Ratio': round(paraphrased_ratio, 3),
            'Sampling_Strategy': sampling_strategy,
            'Target_Met': total_sampled == target_queries_per_specialty
        })
    
    return pd.DataFrame(results)

def generate_actual_samples(df, sampling_results, paraphrased_queries_dict, synthetic_queries_dict):
    """
    Generate actual query samples based on the sampling strategy.
    
    Args:
        df: Original DataFrame with statistics
        sampling_results: Results from sample_queries_per_specialty function
        paraphrased_queries_dict: Dictionary with paraphrased queries by specialty
        synthetic_queries_dict: Dictionary with synthetic queries by specialty
    
    Returns:
        Dictionary with sampled queries by specialty
    """
    sampled_queries = {}
    
    for _, row in sampling_results.iterrows():
        specialty = row['Specialties']
        n_paraphrased = row['Sampled_Paraphrased']
        n_synthetic = row['Sampled_Synthetic']
        
        specialty_samples = {
            'paraphrased': [],
            'synthetic': [],
            'total_count': row['Total_Sampled']
        }
        
        # Sample paraphrased queries
        if n_paraphrased > 0 and specialty in paraphrased_queries_dict:
            available_paraphrased = paraphrased_queries_dict[specialty]
            if len(available_paraphrased) >= n_paraphrased:
                specialty_samples['paraphrased'] = np.random.choice(
                    available_paraphrased, 
                    size=n_paraphrased, 
                    replace=False
                ).tolist()
            else:
                specialty_samples['paraphrased'] = available_paraphrased
        
        # Sample synthetic queries
        if n_synthetic > 0 and specialty in synthetic_queries_dict:
            available_synthetic = synthetic_queries_dict[specialty]
            if len(available_synthetic) >= n_synthetic:
                specialty_samples['synthetic'] = np.random.choice(
                    available_synthetic, 
                    size=n_synthetic, 
                    replace=False
                ).tolist()
            else:
                specialty_samples['synthetic'] = available_synthetic
        
        sampled_queries[specialty] = specialty_samples
    
    return sampled_queries

# Main execution
if __name__ == "__main__":
    # Load your data (replace with your actual data loading)
    paraphrased_production_queries = load_paraphrased_queries()
    
    # Get summary statistics (replace with your actual function call)
    # final_stats = get_summary_stats(specialty_paraphrased_queries_dict, use_specialty_query_dict)
    
    # For demonstration, using sample data based on your screenshot
    sample_data = {
        'Specialties': [
            'acupuncturist_acupuncturist',
            'advanced practice midwife_advanced practice midwife',
            'allergy & immunology_allergy & immunology',
            'allergy & immunology_allergy',
            'allergy & immunology_clinical & laboratory immunology',
            'ambulance_air transport',
            'ambulance_land transport',
            'anesthesiology_addiction medicine',
            'anesthesiology_anesthesiology',
            'anesthesiology_critical care medicine'
        ],
        'Total_Queries_Paraphrased': [0, 5, 30, 80, 30, 0, 0, 75, 5, 55],
        'Total_Queries_Synthetic': [310, 396, 379, 382, 376, 374, 356, 397, 354, 360],
        'Total_Queries': [310, 401, 409, 462, 406, 374, 356, 472, 359, 415]
    }
    
    df = pd.DataFrame(sample_data)
    
    # Apply sampling algorithm
    sampling_results = sample_queries_per_specialty(df, target_queries_per_specialty=250)
    
    # Display results
    print("Sampling Results:")
    print("=" * 80)
    print(sampling_results.to_string(index=False))
    
    # Summary statistics
    print(f"\nSummary:")
    print(f"Total specialties: {len(sampling_results)}")
    print(f"Specialties meeting target (250): {sampling_results['Target_Met'].sum()}")
    print(f"Average paraphrased ratio: {sampling_results['Paraphrased_Ratio'].mean():.3f}")
    
    # Optional: Generate actual samples (uncomment if you have the query dictionaries)
    # sampled_queries = generate_actual_samples(df, sampling_results, paraphrased_queries_dict, synthetic_queries_dict)
    # 
    # # Save results
    # sampling_results.to_csv('sampling_results.csv', index=False)
    # with open('sampled_queries.json', 'w') as f:
    #     json.dump(sampled_queries, f, indent=2)