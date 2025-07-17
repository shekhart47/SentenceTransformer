# Usage Example

# python3 gpt_model_nucc_specialty_prediction_4.py

import os
import json
import time
import openai
import pickle
import warnings
import requests
import pandas as pd
from tqdm import tqdm
from typing import List, Optional, Literal
from pydantic import BaseModel, Field, validator
from azureml.core import Workspace
from azure.identity import DefaultAzureCredential
from azureml.core.authentication import ServicePrincipalAuthentication
from langchain.chains import LLMChain
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel as LangChainBaseModel, Field as LangChainField
from langchain.output_parsers import PydanticOutputParser, CommaSeparatedListOutputParser, JsonOutputKeyToolsParser

# Suppress warnings

pd.set_option(‘display.max_rows’, None)
warnings.filterwarnings(“ignore”)

# Azure ML Workspace setup

ws = Workspace.from_config()

class BearerAuth(requests.auth.AuthBase):
def **init**(self, token):
self.token = token

```
def __call__(self, r):
    r.headers["authorization"] = "Bearer " + self.token
    return r
```

def initialize_llm(model_name) -> AzureChatOpenAI:
“”“Initialize Azure OpenAI model with proper configuration”””
ws = Workspace.from_config()
keyvault = ws.get_default_keyvault()
workspace_name = keyvault.get_secret(“project-workspace-name”)
access_token = keyvault.get_secret(“https://cognitiveservices.azure.com/.default”)

```
# Environment setup for different models
if model_name == "gpt-4o":
    os.environ["AZURE_OPENAI_API_VERSION"] = "2024-05-01-preview"
    os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = "gpt-4o"
elif model_name == "gpt-4.1":
    os.environ["AZURE_OPENAI_API_VERSION"] = "2024-12-01-preview"
    os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = "gpt-4.1"

# Initialize model
model = AzureChatOpenAI(
    deployment_name=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    azure_endpoint=os.environ["AZURE_OPENAI_API_BASE"],
    openai_api_type="azure_ad",
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
    max_tokens=4000,
    temperature=0.0,
    model_kwargs={"seed": 1337}
)

print(f'Model {model_name} initialized')
return model
```

# Define Pydantic models for strict validation

class SpecialtyPrediction(BaseModel):
“”“Response model for specialty prediction with strict validation”””
specialty: str = Field(description=“Medical specialty from the provided list”)
confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description=“Confidence score (0-1)”)

```
@validator('specialty')
def validate_specialty_in_list(cls, v, values, **kwargs):
    """This validator will be enhanced in the function to check against the actual list"""
    return v
```

class SpecialtiesResponse(BaseModel):
“”“Response model for multiple specialties”””
specialties: List[str] = Field(description=“List of medical specialties from the provided list”)

```
@validator('specialties')
def validate_all_specialties_in_list(cls, v, values, **kwargs):
    """This validator will be enhanced in the function to check against the actual list"""
    return v
```

def create_dynamic_specialty_model(specialty_list: List[str]):
“”“Create a dynamic Pydantic model with enum validation for specialties”””

```
# Create a literal type from the specialty list
SpecialtyEnum = Literal[tuple(specialty_list)]

class DynamicSpecialtyPrediction(BaseModel):
    specialty: SpecialtyEnum = Field(description=f"Medical specialty that must be one of: {', '.join(specialty_list[:5])}...")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Confidence score (0-1)")

class DynamicSpecialtiesResponse(BaseModel):
    specialties: List[SpecialtyEnum] = Field(description=f"List of medical specialties that must be from: {', '.join(specialty_list[:5])}...")

return DynamicSpecialtyPrediction, DynamicSpecialtiesResponse
```

def get_specialty_for_query_nucc_labels(
model: AzureChatOpenAI,
specialty_list: List[str],
user_query: str,
use_strict_validation: bool = True
) -> str:
“””
Get medical specialty prediction with strict validation

```
Args:
    model: Azure OpenAI model instance
    specialty_list: List of valid medical specialties
    user_query: User's search query
    use_strict_validation: Whether to use strict Pydantic validation

Returns:
    Predicted specialty from the provided list
"""

if use_strict_validation:
    # Create dynamic model with enum validation
    SpecialtyPrediction, _ = create_dynamic_specialty_model(specialty_list)
    output_parser = PydanticOutputParser(pydantic_object=SpecialtyPrediction)
else:
    # Use the base model
    output_parser = PydanticOutputParser(pydantic_object=SpecialtyPrediction)

# Enhanced system prompt with explicit instructions
system_prompt = f"""You are a helpful AI assistant specializing in healthcare.
```

Your task is to identify the medical specialty and sub-specialty associated with a user provided search query.

CRITICAL INSTRUCTIONS:

1. You MUST only return specialties from this exact list: {specialty_list}
1. Find the closest match from the provided list
1. If no close match exists, return the most general applicable specialty from the list
1. NEVER create new specialties or modify the provided specialty names
1. The specialty name must match EXACTLY as provided in the list

Available specialties: {’, ’.join(specialty_list)}

Examples:

- Query: “long term facility” → Look for geriatrics, physical therapy, or general practice from the list
- Query: “comfort care” → Look for geriatrics, general practice, or relevant specialty from the list

{output_parser.get_format_instructions()}

Remember: ONLY use specialties from the provided list. If uncertain, choose the most relevant general specialty from the list.
“””

```
# Create prompt template
prompt_template = PromptTemplate(
    template=system_prompt + "\n\nUser Query: {user_query}",
    input_variables=["user_query"],
    partial_variables={"format_instructions": output_parser.get_format_instructions()}
)

# Create chain
chain = prompt_template | model | output_parser

try:
    # Get prediction
    result = chain.invoke({"user_query": user_query})
    
    # Additional validation layer
    if hasattr(result, 'specialty'):
        predicted_specialty = result.specialty
    else:
        predicted_specialty = str(result)
    
    # Fallback validation - ensure the specialty is in the list
    if predicted_specialty not in specialty_list:
        # Find closest match using fuzzy matching
        closest_match = find_closest_specialty(predicted_specialty, specialty_list)
        print(f"Warning: Model returned '{predicted_specialty}' not in list. Using closest match: '{closest_match}'")
        return closest_match
    
    return predicted_specialty
    
except Exception as e:
    print(f"Error in prediction: {e}")
    # Return a default specialty from the list
    return specialty_list[0] if specialty_list else "general_practice"
```

def find_closest_specialty(predicted: str, specialty_list: List[str]) -> str:
“”“Find the closest matching specialty using simple string similarity”””
predicted_lower = predicted.lower()

```
# Direct substring matching first
for specialty in specialty_list:
    if predicted_lower in specialty.lower() or specialty.lower() in predicted_lower:
        return specialty

# Keyword-based matching
predicted_words = set(predicted_lower.split())
best_match = specialty_list[0]
best_score = 0

for specialty in specialty_list:
    specialty_words = set(specialty.lower().split())
    common_words = predicted_words.intersection(specialty_words)
    score = len(common_words) / max(len(predicted_words), len(specialty_words))
    
    if score > best_score:
        best_score = score
        best_match = specialty

return best_match
```

def load_datasets() -> tuple:
“”“Load and combine specialty datasets”””
# Load JSON datasets
path_1 = ‘../../../datasets/datasets_error_analysis/result_evaluation_inhouse_gpt/gpt_specialty_query_dict_1.json’
path_2 = ‘../../../datasets/datasets_error_analysis/result_evaluation_inhouse_gpt/gpt_specialty_query_dict_2.json’
path_3 = ‘../../../datasets/datasets_error_analysis/result_evaluation_inhouse_gpt/gpt_specialty_query_dict_3.json’
path_4 = ‘../../../datasets/datasets_error_analysis/result_evaluation_inhouse_gpt/gpt_specialty_query_dict_4.json’

```
# Load data
datasets = []
for path in [path_1, path_2, path_3, path_4]:
    with open(path, 'r') as file:
        datasets.append(json.load(file))

# Combine data
gpt_combined_data = {}
for data in datasets:
    for specialty, queries in data.items():
        if specialty in gpt_combined_data:
            gpt_combined_data[specialty].update(queries)
        else:
            gpt_combined_data[specialty] = queries.copy()

all_specialties = list(gpt_combined_data.keys())
print(f'Total Specialties Covered: {len(all_specialties)}')

return all_specialties
```

def load_ues_keywords_list():
“”“Load UES keywords dataset”””
path = ‘../../../datasets/UES_Keywords/ues_keywords_part2.csv’
ues_keywords = pd.read_csv(path)
data_set2 = list(ues_keywords[‘Keywords’])

```
# Load additional keywords
with open('../../../datasets/UES_Keywords/ues_keywords_in_api_from_audit_file_20250408_1 (1).json', 'r') as file:
    data_set1 = json.load(file)

# Combine datasets
final_keywords = data_set2 + data_set1
final_keywords = pd.DataFrame(list(zip(final_keywords)), columns=['final_keywords'])
final_keywords['sequence_length'] = final_keywords['final_keywords'].apply(lambda x: len(x.split()))

return final_keywords
```

def main():
“”“Main execution function”””
# Initialize model
model_name = “gpt-4.1”
model = initialize_llm(model_name=model_name)

```
# Load datasets
nucc_specialties = load_datasets()
final_keywords = load_ues_keywords_list()
query_list = final_keywords['final_keywords']

print(f"Number of queries to process: {len(final_keywords)}")

# Initialize results dictionary
query_specialty_dict = {}

# Process queries with progress bar
for i in tqdm(range(len(query_list))):
    user_query = query_list[i]
    
    # Get prediction with strict validation
    predictions = get_specialty_for_query_nucc_labels(
        model=model, 
        specialty_list=nucc_specialties, 
        user_query=user_query,
        use_strict_validation=True
    )
    
    query_specialty_dict[user_query] = predictions
    time.sleep(0.1)  # Rate limiting

# Save results
output_path = '../../../datasets/datasets_augmented/augmentation_set3/ues_keyword_nucc_classification/ues_keyword_nucc_classification.json'
with open(output_path, 'w') as file:
    json.dump(query_specialty_dict, file, indent=4)

print(f"Results saved to: {output_path}")
print(f"Processed {len(query_specialty_dict)} queries")
```

if **name** == “**main**”:
main()
