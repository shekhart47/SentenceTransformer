def consolidate_icd_datasets(data_list_set1, data_list_set2):
“””
Consolidates two datasets of medical specialty dictionaries by merging
ICD code descriptions for matching specialties and queries.

```
Args:
    data_list_set1: List of dictionaries with medical specialty data
    data_list_set2: List of dictionaries with additional/missing ICD descriptions

Returns:
    List of consolidated dictionaries with merged ICD descriptions
"""

# Convert lists to dictionaries for easier lookup
def list_to_dict(data_list):
    """Convert list of dictionaries to single dictionary by specialty key"""
    result = {}
    for item in data_list:
        for specialty_key, queries_dict in item.items():
            if specialty_key in result:
                # If specialty already exists, merge the queries
                for query, icd_codes in queries_dict.items():
                    if query in result[specialty_key]:
                        # Extend existing ICD codes list, avoiding duplicates
                        existing_codes = set(result[specialty_key][query])
                        new_codes = set(icd_codes)
                        result[specialty_key][query] = list(existing_codes.union(new_codes))
                    else:
                        result[specialty_key][query] = icd_codes
            else:
                result[specialty_key] = queries_dict.copy()
    return result

# Convert both datasets to dictionaries
dict_set1 = list_to_dict(data_list_set1)
dict_set2 = list_to_dict(data_list_set2)

# Start with set1 as base
consolidated = dict_set1.copy()

# Merge data from set2
for specialty_key, queries_dict in dict_set2.items():
    if specialty_key in consolidated:
        # Specialty exists in both sets
        for query, icd_codes in queries_dict.items():
            if query in consolidated[specialty_key]:
                # Query exists in both - merge ICD codes
                existing_codes = consolidated[specialty_key][query]
                
                # If existing codes are empty and new codes are available
                if not existing_codes and icd_codes:
                    consolidated[specialty_key][query] = icd_codes
                # If existing codes exist and new codes also exist, merge them
                elif existing_codes and icd_codes:
                    existing_set = set(existing_codes)
                    new_set = set(icd_codes)
                    consolidated[specialty_key][query] = list(existing_set.union(new_set))
                # If both are empty or only existing has codes, keep existing
            else:
                # Query doesn't exist in set1, add it from set2
                consolidated[specialty_key][query] = icd_codes
    else:
        # Specialty doesn't exist in set1, add entire specialty from set2
        consolidated[specialty_key] = queries_dict.copy()

# Convert back to list format
result_list = []
for specialty_key, queries_dict in consolidated.items():
    result_list.append({specialty_key: queries_dict})

return result_list
```

def print_consolidation_summary(original_set1, original_set2, consolidated):
“””
Print a summary of the consolidation process
“””
def count_entries(data_list):
specialty_count = 0
query_count = 0
empty_queries = 0

```
    for item in data_list:
        for specialty_key, queries_dict in item.items():
            specialty_count += 1
            for query, icd_codes in queries_dict.items():
                query_count += 1
                if not icd_codes:
                    empty_queries += 1
    
    return specialty_count, query_count, empty_queries

set1_specs, set1_queries, set1_empty = count_entries(original_set1)
set2_specs, set2_queries, set2_empty = count_entries(original_set2)
cons_specs, cons_queries, cons_empty = count_entries(consolidated)

print("=== Consolidation Summary ===")
print(f"Original Set 1: {set1_specs} specialties, {set1_queries} queries, {set1_empty} empty")
print(f"Original Set 2: {set2_specs} specialties, {set2_queries} queries, {set2_empty} empty")
print(f"Consolidated:   {cons_specs} specialties, {cons_queries} queries, {cons_empty} empty")
print(f"Reduction in empty queries: {(set1_empty + set2_empty) - cons_empty}")
```

def find_empty_queries(data_list):
“””
Find and return queries with empty ICD code descriptions
“””
empty_queries = []

```
for item in data_list:
    for specialty_key, queries_dict in item.items():
        for query, icd_codes in queries_dict.items():
            if not icd_codes:
                empty_queries.append({
                    'specialty': specialty_key,
                    'query': query
                })

return empty_queries
```

# Example usage:

if **name** == “**main**”:
# Your data would be loaded here
# data_list_set1 = … (your first dataset)
# data_list_set2 = … (your second dataset)

```
# Consolidate the datasets
consolidated_data = consolidate_icd_datasets(data_list_set1, data_list_set2)

# Print summary
print_consolidation_summary(data_list_set1, data_list_set2, consolidated_data)

# Check remaining empty queries
remaining_empty = find_empty_queries(consolidated_data)
if remaining_empty:
    print(f"\nRemaining empty queries: {len(remaining_empty)}")
    for item in remaining_empty[:5]:  # Show first 5
        print(f"  - {item['specialty']}: {item['query']}")
    if len(remaining_empty) > 5:
        print(f"  ... and {len(remaining_empty) - 5} more")
else:
    print("\nNo empty queries remaining!")

# Save consolidated data if needed
# import json
# with open('consolidated_icd_data.json', 'w') as f:
#     json.dump(consolidated_data, f, indent=2)
```
