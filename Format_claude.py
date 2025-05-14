I understand your requirement. You need to process a list of dictionaries containing medical query information and group them by a key called "keyed_specialty", which is derived from combining the "nucc_class" and "nucc_spec" fields. Let me write a function that accomplishes this:

```python
from collections import defaultdict

def group_queries_by_specialty(dataset):
    """
    Group queries and ICD codes by keyed_specialty.
    
    Args:
        dataset: List of dictionaries containing medical query information
        
    Returns:
        Dictionary where key is keyed_specialty and value is a dictionary with:
            - 'queries': List of unique queries for that specialty
            - 'icd_codes': List of unique ICD codes for that specialty
    """
    specialty_dict = defaultdict(lambda: {'queries': [], 'icd_codes': []})
    
    for data_temp in dataset:
        # Extract the required fields
        query = data_temp.get('query', '')
        nucc_spec = data_temp.get('nucc_spec', '').lower()
        nucc_class = data_temp.get('nucc_class', '').lower()
        icd_codes = data_temp.get('related_icd10cm', [])
        
        # Create the keyed_specialty
        keyed_specialty = nucc_class + '_' + nucc_spec
        
        # Replace any '/' characters
        if '/' in keyed_specialty:
            keyed_specialty = keyed_specialty.replace('/', '')
        
        # Add the query and ICD codes to the appropriate specialty
        if query and query not in specialty_dict[keyed_specialty]['queries']:
            specialty_dict[keyed_specialty]['queries'].append(query)
        
        # Add any new ICD codes to the specialty's list
        for icd in icd_codes:
            if icd not in specialty_dict[keyed_specialty]['icd_codes']:
                specialty_dict[keyed_specialty]['icd_codes'].append(icd)
    
    # Convert defaultdict to regular dict for return
    return dict(specialty_dict)
```

This function will:

1. Create a defaultdict to store the grouped data
2. Iterate through each dictionary in your dataset
3. Extract the required fields (query, nucc_spec, nucc_class, icd_codes)
4. Create the keyed_specialty by combining nucc_class and nucc_spec with an underscore
5. Replace any '/' characters in the keyed_specialty
6. Add the query and ICD codes to the appropriate specialty's lists, ensuring no duplicates
7. Return the result as a regular dictionary

You can use it like this:

```python
# Example usage
result = group_queries_by_specialty(gemini_dataset)

# To check the result for a specific specialty
if 'allergy & immunology_allergy' in result:
    specialty_data = result['allergy & immunology_allergy']
    print(f"Queries: {specialty_data['queries']}")
    print(f"ICD Codes: {specialty_data['icd_codes']}")
```

Based on your example, the query "Nickel allergy rash" would be grouped under the key "allergy & immunology_allergy" along with its associated ICD codes "L23.0" and "Z91.048".

Does this match the functionality you need? I can modify the solution if you need additional features or have different requirements for the output format.​​​​​​​​​​​​​​​​
