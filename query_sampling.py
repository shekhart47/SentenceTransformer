import pandas as pd
import numpy as np
import json

def get_synthetic_queries():
    """Load synthetic queries from JSON files"""
    path = '../../../datasets/datasets_augmented/augmentation_set4/iteration1/'
    all_files = [path + file for file in os.listdir(path) if '.json' in file]
    
    specialty_query_dict = {}
    for file_path in all_files:
        with open(file_path, 'r') as file:
            data_specialty = json.load(file)
            
        specialty = list(data_specialty.keys())[0]
        queries = list(data_specialty.values())[0]
        
        specialty_query_dict[specialty] = queries
    
    return specialty_query_dict

def get_paraphrased_production_queries():
    """Load paraphrased queries from JSON file"""
    file_path = './specialty_paraphrased_queries_dict.json'
    
    with open(file_path, 'r') as file:
        paraphrased_production_queries = json.load(file)
    
    return paraphrased_production_queries

def get_summary_stats(specialty_query_dict, paraphrased_production_queries):
    """
    Generate summary statistics based on the loaded query dictionaries.
    Uses specialty_query_dict keys as reference since it has more specialties.
    """
    stats_data = []
    
    for specialty in specialty_query_dict.keys():
        # Count synthetic queries
        synthetic_count = len(specialty_query_dict[specialty])
        
        # Count paraphrased queries (may not exist for all specialties)
        paraphrased_count = len(paraphrased_production_queries.get(specialty, []))
        
        # Total queries
        total_count = synthetic_count + paraphrased_count
        
        stats_data.append({
            'Specialties': specialty,
            'Total_Queries_Paraphrased': paraphrased_count,
            'Total_Queries_Synthetic': synthetic_count,
            'Total_Queries': total_count
        })
    
    return pd.DataFrame(stats_data)

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

def generate_actual_samples(sampling_results, paraphrased_production_queries, specialty_query_dict):
    """
    Generate actual query samples based on the sampling strategy.
    
    Args:
        sampling_results: Results from sample_queries_per_specialty function
        paraphrased_production_queries: Dictionary with paraphrased queries by specialty
        specialty_query_dict: Dictionary with synthetic queries by specialty
    
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
        if n_paraphrased > 0 and specialty in paraphrased_production_queries:
            available_paraphrased = paraphrased_production_queries[specialty]
            if len(available_paraphrased) >= n_paraphrased:
                specialty_samples['paraphrased'] = np.random.choice(
                    available_paraphrased, 
                    size=n_paraphrased, 
                    replace=False
                ).tolist()
            else:
                specialty_samples['paraphrased'] = available_paraphrased
        
        # Sample synthetic queries
        if n_synthetic > 0 and specialty in specialty_query_dict:
            available_synthetic = specialty_query_dict[specialty]
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
    # Load your actual data
    specialty_query_dict = get_synthetic_queries()
    paraphrased_production_queries = get_paraphrased_production_queries()
    
    # Get summary statistics using your actual data
    final_stats = get_summary_stats(specialty_query_dict, paraphrased_production_queries)
    
    print(f"Loaded data:")
    print(f"- Total specialties in synthetic data: {len(specialty_query_dict)}")
    print(f"- Total specialties in paraphrased data: {len(paraphrased_production_queries)}")
    print(f"- Specialties with both types: {len(set(specialty_query_dict.keys()) & set(paraphrased_production_queries.keys()))}")
    print(f"- Specialties with only synthetic: {len(set(specialty_query_dict.keys()) - set(paraphrased_production_queries.keys()))}")
    print()
    
    # Apply sampling algorithm
    sampling_results = sample_queries_per_specialty(final_stats, target_queries_per_specialty=250)
    
    # Display results
    print("Sampling Results:")
    print("=" * 120)
    print(sampling_results.to_string(index=False))
    
    # Summary statistics
    print(f"\nSummary:")
    print(f"Total specialties: {len(sampling_results)}")
    print(f"Specialties meeting target (250): {sampling_results['Target_Met'].sum()}")
    print(f"Average paraphrased ratio: {sampling_results['Paraphrased_Ratio'].mean():.3f}")
    print(f"Specialties with no paraphrased queries: {(sampling_results['Original_Paraphrased'] == 0).sum()}")
    print(f"Specialties with >200 paraphrased queries: {(sampling_results['Original_Paraphrased'] > 200).sum()}")
    
    # Generate actual samples
    print("\nGenerating actual query samples...")
    sampled_queries = generate_actual_samples(sampling_results, paraphrased_production_queries, specialty_query_dict)
    
    # Save results
    print("\nSaving results...")
    sampling_results.to_csv('sampling_results.csv', index=False)
    with open('sampled_queries.json', 'w') as f:
        json.dump(sampled_queries, f, indent=2)
    
    print("Results saved to:")
    print("- sampling_results.csv: Detailed sampling statistics")
    print("- sampled_queries.json: Actual sampled queries by specialty")
    
    # Display sample of actual queries for verification
    print(f"\nSample verification (first specialty):")
    first_specialty = list(sampled_queries.keys())[0]
    sample_data = sampled_queries[first_specialty]
    print(f"Specialty: {first_specialty}")
    print(f"- Paraphrased samples: {len(sample_data['paraphrased'])}")
    print(f"- Synthetic samples: {len(sample_data['synthetic'])}")
    print(f"- Total samples: {sample_data['total_count']}")
    if sample_data['paraphrased']:
        print(f"- First paraphrased query: {sample_data['paraphrased'][0][:100]}...")
    if sample_data['synthetic']:
        print(f"- First synthetic query: {sample_data['synthetic'][0][:100]}...")