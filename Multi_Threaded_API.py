
import os
import time
import json
import openai
import pickle
import requests
import itertools
import warnings
import pandas as pd
from tqdm import tqdm
from typing import List
from collections import defaultdict

from azureml.core import Workspace
from azureml.core.authentication import ServicePrincipalAuthentication

from langchain.chains import LLMChain
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain.output_parsers import PydanticOutputParser, JsonOutputKeyToolsParser, CommaSeparatedListOutputParser

# Filter warnings
warnings.filterwarnings("ignore")

# Configure workspace
ws = Workspace.from_config()

class BearerAuth(requests.auth.AuthBase):
    def __init__(self, token):
        self.token = token
        
    def __call__(self, r):
        r.headers["Authorization"] = "Bearer " + self.token
        return r

def initialize_llm() -> AzureChatOpenAI:
    ws = Workspace.from_config()
    keyvault = ws.get_default_keyvault()
    credential = ServicePrincipalAuthentication()
    workspacename = keyvault.get_secret("project-workspace-name")
    access_token = credential.get_token("https://cognitiveservices.azure.com/.default")
    os.environ["AZURE_OPENAI_KEY"] = access_token.token
    openai.api_type = "azure_ad"
    os.environ["AZURE_OPENAI_ENDPOINT"] = f"https://{workspacename}openai.openai.azure.com/"
    openai.api_base = os.environ["AZURE_OPENAI_ENDPOINT"]
    subscription_id = keyvault.get_secret("project-subscription-id")
    
    # Ensure you have these environment variables set up with your Azure OpenAI credentials
    os.environ["AZURE_OPENAI_API_KEY"] = "0eddd6654bd4427ba4f5580b5a0db0a"
    os.environ["AZURE_OPENAI_API_BASE"] = "https://xgroj1mb2vjlooopenai.openai.azure.com/"
    os.environ["AZURE_OPENAI_API_VERSION"] = "2024-05-01-preview"
    os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = "gpt-4o"
    
    subscriptionId = keyvault.get_secret("project-subscription-id")
    apiVersion = "2023-08-01-preview"
    url = f"https://management.azure.com/subscriptions/{subscriptionId}/resourceGroups/{workspacename}-common/providers/Microsoft.CognitiveServices/accounts/{workspacename}openai/deployments?api-version={apiVersion}"
    accessToken = credential.get_token("https://management.azure.com/.default")
    response = requests.get(url, auth=BearerAuth(accessToken.token))
    
    print(f'Initializing Model : {os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]}')
    model = AzureChatOpenAI(
        deployment_name=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
        azure_endpoint=os.environ["AZURE_OPENAI_API_BASE"],
        openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
        max_tokens=4000,
        temperature=0.9,
        model_kwargs={"seed": 1337}
    )
    
    print('Model Initialized')
    return model

def get_query_icd_codes_v4(model: AzureChatOpenAI, target_specialty: str, medical_query: str):
    # This version of the prompt can handle queries with location based information added in
    # eg:
    # 1 - arm pain doctor in redmond, washington : list of ICD codes
    # 2 - Dr Sara Moore : no results
    # 3 - Dr Sara Moore in Redmond, Washington : no results
    # 4 - Dr Sara Moore physiotherapist in Redmond, Walington : list of ICD codes
    
    class SpecialiesResponse(BaseModel):
        queries: List[str] = Field(description="List of queries corresponding to user provided medical specialty")
    
    # Set up the PydanticOutputParser with the SpecialiesResponse model
    output_parser = CommaSeparatedListOutputParser()
    
    system_prompt = """You are a certified medical coder specializing in ICD-10 code assignment for search queries.
Your task is to generate correct and relevant ICD-10 codes based on a medical query and target_specialty provided by the user.

Please note that the target specialty provided by the user contains a specialty and a subspecialty as follows : specialty_subspecialty. 
For example if the specialty is internal medicine and subspecialty is endocrinology then the user input would be internal medicine_endocrinology

Please develop an understanding of the intent of the user provided medical search query with respect to the target_specialty 
and adhere to the following guidelines to generate the ICD codes:

Strict Guidelines:
- Return only valid ICD-10 codes separated by comma , without additional text.
- Generate upto 10 ICD-10 codes that are most relevant to the target_specialty for the identified medical terms
- If there are no ICD codes for the medical query, return "[]" to indicate an empty list.

Query Analysis Rules:
1. If the query contains ONLY names or locations without medical terms example santa monica, bellevue, return "[]".
2. If the query contains the target_specialty itself (e.g. "cardiologist" for cardiology) generate top relevant ICD-10 codes for the target specialty.
3. If the query contains symptoms/procedures related to the target_specialty, generate top relevant ICD-10 codes for the search query for that target specialty
4. For any query mentioning a medical profession role (e.g. "physiotherapist","cardiologist","oncologist", psychiatrist etc.), analyze role and target_specialty 
   and generate common ICD-10 codes for conditions typically treated by that specialty.
     Example : sanata monica orthopedics group,instruction : Extract orthopedics from the query and generate ICD-10 codes for that
5. Always prioritize specific medical conditions mentioned in the query (e.g. "knee pain","back pain") over generic specialty terms.
6. If the query contains both a specialty term AND a specific condition, prioritize codes for the specific condition.
7. Analyze the search query and extract relevant tokens from the search query which can be mapped to ICD-10 codes, after extraction return the relevant ICD-10 codes

To help you understand the queries with the strict guideline 1, I am providing the following examples:

Example 1 : 
Medical Query : Early pregnancy signs
Target Specialty : Gynecology  
ICD_CODES : Z34.01, Z34.81, Z34.81
Reason For ICD codes : The query is medical in nature and when analyzed from the perspective of the target specialty Gynecology can be associated with the above listed ICD Codes.

Example 2 : 
Medical Query : chest pain
Target Specialty : Cardiology
ICD_CODES : R07.9, I20.9, R07.89, I21.3,I25.10, R07.1, R07.2, I351.0
Reason For ICD codes : The query is medical in nature and when analyzed from the perspective of the target specialty Cardiology can be associated with the above listed ICD Codes.

Example 3 : 
Medical Query : Sara Moore
Target Specialty : Neurology
ICD_CODES : []
Reason for ICD codes : The seach query contains no medical term(s) and thus as per the Strict Guidelines should not have medical codes 

Example 4 : 
Medical Query : Physical Therapist near Highland Ave
Target Specialty : Physiotherapy
ICD_CODES : [M54.5, M25.50, M62.81,S33, S3XA, M79.1, M62.830, M54.2, M54.16, Z96.641, Z47.1]
Reason for ICD codes : Some parts of the search query are medical in nature, and thus from the search query we can extract 'Physical Therapist' and relate it with the
target specialty Physiotherapy to generate top relevant ICD Codes for the extracted terms

Example 5 : 
Medical Query : Dr Smith Orthopedic surgeon knee replacement
Target Specialty : Orthopedics
ICD_CODES : [M17.0, M17.11, M17.12, Z96.651, Z96.652, Z96.653, Z47.1, M25.561, M25.562, M79.604]
Reason for ICD codes : Some parts of the search query are medical in nature, and thus from the search query we can extract 'Orthopedic surgeon knee replacement' and related it with
the target specialty Orthopedics to generate top relevant ICD Codes for the extracted terms

Example 6 : 
Medical Query : Headache Specialist
Target Specialty : Neurology
ICD_CODES : [G44.309, R51.0, G44.009, G44.319, G43.709, G44.89, R22.0, G44.019, G44.809]
Reason for ICD codes : The query is medical in nature, and thus we can relate it with the target specialty Neurology to generate top relevant ICD Codes for the search query

Example 7 : 
Medical Query : James Young physiotherapist in Baltimore for back pain
Target Specialty : Physiotherapist
ICD_CODES : [M54.5, M54.4, M54.16, M51.26, M51.27, M47.26, M47.27, M47.28, M54.89, M54.9]
Reason for ICD codes : Some parts of the search query are medical in nature, and thus from the search query we can extract 'physiotherapist', 'back pain' and related them with 
the target specialty Physiotherapist to generate top relevant ICD Codes for the extracted terms

Example 8 : 
Medical Query : Santa Monica Orthopedics group
Target Specialty : Orthopedics
ICD_CODES : [M17.0, M17.11, M17.12, M25.561, M25.562, M23.50, M79.604]
Reason for ICD codes : Some parts of the search query are medical in nature, and thus from the search query we can extract Orthopedics and related it with the target 
specialty Orthopedics to generate top relevant ICD Codes for the extracted terms

Example 9 : 
Medical Query : Blood Lab
Target Specialty : pathology_blood banking & transfusion medicine
ICD_CODES : ['D50.9', 'D64', 'D69.6', 'D73.81', 'D73.89', 'D73.9', 'Z13.0', 'Z31.812']
Reason for ICD codes : Some parts of the search query are medical in nature, and thus from the search query we extracted Blood and related it with the target 
specialty pathology_blood banking & transfusion medicine to generated top relevant ICD Codes for the extracted terms

IMPORTANT:
PLEASE TAKE YOUR TIME IN UNDERSTANDING THE MEDICAL QUERY WITH RESPECT TO target specialty. Your response must contain
ONLY THE ICD-10 codes separated by commas, or "[]" if no codes apply.

Do not include any explanations, headers, or additional text in your response.
PLEASE DO NOT DEVIATE FROM THE ABOVE ASSUMPTION.

Task:
Identify the correct ICD codes for the following medical query.

Format Instructions:
{format_instructions}

medical_query: {medical_query}
target_specialty: {target_specialty}
"""
    
    # Define the prompt template
    prompt_template = PromptTemplate.from_template(
        template=system_prompt,
        partial_variables={"format_instructions": output_parser.get_format_instructions()}
    )
    
    chain = LLMChain(prompt=prompt_template, llm=model, output_parser=output_parser)
    result = chain.invoke(inputs={"target_specialty": target_specialty, "medical_query": medical_query, "format_instructions": output_parser.get_format_instructions()})
    return result

def load_datasets(file_path):
    production_dataset_icd = pd.read_csv('../gpt-4o-augmented-data/Medical_Term_Medical_Code.csv').iloc[:,1:]
    production_dataset_queries = pd.read_csv('../gpt-4o-augmented-data/filtered_specialty_query_list.csv').iloc[:,1:]
    
    with open(file_path, 'r') as file:
        multilabel_specialty_data = json.load(file)
    
    icd_reference = pd.read_csv('../../../dataset/icd10.csv').iloc[:,1:]
    icd_reference = icd_reference.iloc[:,13:15] # 
    icd_reference.columns = ['codes','description']
    icd_reference = icd_reference.drop_duplicates()
    
    icd_reference_lookup = {}
    for row in icd_reference.itertuples():
        icd_reference_lookup[row.codes] = row.description
    
    return production_dataset_icd, production_dataset_queries, icd_reference_lookup, multilabel_specialty_data

def get_query_positives(icd_reference_lookup, multilabel_specialty_data, chunk_index=0, total_chunks=1, use_parallelism=True, batch_size=50, max_workers=4):
    """Process queries by specialty with parallel execution and chunking"""
    # Initialize the LLM
    model = initialize_llm()
    
    specialty_anchor_positives = defaultdict(dict)
    
    # Path where the datasets for each specialty would be saved
    prefix = '../gpt_triplet_data_v5/gpt_based_positives_set_3/'
    os.makedirs(prefix, exist_ok=True)
    
    # Get a list of all specialties we need to build dataset for
    all_specialties = list(multilabel_specialty_data.keys())
    
    # Split specialties based on chunk index and total chunks
    chunk_size = len(all_specialties) // total_chunks + (1 if len(all_specialties) % total_chunks else 0)
    start_idx = chunk_index * chunk_size
    end_idx = min(start_idx + chunk_size, len(all_specialties))
    
    # We iterate over the specialties assigned to this chunk to build separate dataset for each
    for j in tqdm(range(start_idx, end_idx)):
        target_specialty = all_specialties[j]
        
        # Process the specialty to remove any '/' in target_specialty name
        if '/' in target_specialty:
            target_specialty = target_specialty.replace('/', '_')
        
        # Save path of the file for the selected target_specialty
        current_file_path = f'{prefix}{target_specialty}.json'
        
        # Check if the current file we're processing has already been processed
        all_files = [prefix + file for file in os.listdir(prefix) if '.json' in file]
        
        # If the current specialty has already been processed
        if current_file_path in all_files or current_file_path.lower() in all_files:
            print(f'{target_specialty} already processed')
            continue
        
        # If the specialty has not been processed, we get the queries for the specialty and process them one by one
        print(f'Processing {target_specialty}')
        queries = multilabel_specialty_data.get(target_specialty)
        
        if queries is None or len(queries) == 0:
            continue
        
        query_code_dict = {}
        
        # Configure parallel processing if enabled
        if use_parallelism and len(queries) > batch_size:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            # Process in batches
            for batch_start in range(0, len(queries), batch_size):
                batch_end = min(batch_start + batch_size, len(queries))
                batch_queries = queries[batch_start:batch_end]
                
                def process_query(query):
                    try:
                        result = get_query_icd_codes_v4(model=model, target_specialty=target_specialty, medical_query=query)
                        
                        # Parse the response - assuming result is a comma-separated string of ICD codes
                        if isinstance(result, str):
                            generated_codes = result.split(',')
                        else:
                            # Handle case when result is already a list or other structure
                            generated_codes = result
                            
                        # Add a layer to verify the codes is in reference table, and get the descriptions of the codes
                        filtered_codes_descriptions = {icd_reference_lookup.get(code.strip(), "") for code in generated_codes if code.strip() in icd_reference_lookup}
                        
                        return query, filtered_codes_descriptions
                    except Exception as e:
                        print(f"Error Processing Label for {query}: {error}" , e)
                        return query, set()
                
                # Execute batch in parallel
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_query = {executor.submit(process_query, query): query for query in batch_queries}
                    
                    for future in as_completed(future_to_query):
                        query, filtered_codes = future.result()
                        query_code_dict[query] = list(filtered_codes)
                
                # Save progress after each batch
                specialty_anchor_positives[target_specialty] = query_code_dict
                file_path = f'{prefix}{target_specialty}.json'
                with open(file_path, 'w') as json_file:
                    json.dump(specialty_anchor_positives, json_file, indent=4)  # Indent is optional, but makes the json file more readable
        
        else:
            # Sequential processing for small number of queries
            for i in tqdm(range(len(queries))):
                medical_query = queries[i]
                try:
                    generated_codes = get_query_icd_codes_v4(model=model, target_specialty=target_specialty, medical_query=medical_query)
                    
                    # Parse the response - assuming result is a comma-separated string of ICD codes
                    if isinstance(generated_codes, str):
                        generated_codes = generated_codes.split(',')
                    
                    # Add a layer to verify the codes is in reference table, and get the descriptions of the codes
                    filtered_codes_descriptions = [icd_reference_lookup.get(code.strip(), "") for code in generated_codes if code.strip() in icd_reference_lookup]
                    
                    query_code_dict[medical_query] = filtered_codes_descriptions
                except Exception as e:
                    print(f"Error Producing Label for {medical_query}: {error}", e)
                    continue
            
            # Save progress after all queries in this specialty
            specialty_anchor_positives[target_specialty] = query_code_dict
            file_path = f'{prefix}{target_specialty}.json'
            with open(file_path, 'w') as json_file:
                json.dump(specialty_anchor_positives, json_file, indent=4)  # Indent is optional, but makes the json file more readable
        
        # Reinitialize the dictionary again to have fresh set of predictions saved as JSON
        specialty_anchor_positives = defaultdict(dict)
    
    return specialty_anchor_positives

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate ICD codes for medical queries')
    parser.add_argument('--chunk', type=int, default=0, help='Chunk index to process')
    parser.add_argument('--total_chunks', type=int, default=1, help='Total number of chunks')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size for parallel processing')
    parser.add_argument('--workers', type=int, default=4, help='Number of worker threads')
    parser.add_argument('--no_parallel', action='store_true', help='Disable parallel processing')
    args = parser.parse_args()
    
    print(f'Processing Chunk {args.chunk} of {args.total_chunks}')
    file_path = '../gpt_triplet_data_v5/gpt_based_positives_set_3/splits/gpt_specialties_split_{args.chunk}.json'
    
    # Load datasets
    _, icd_reference_lookup, multilabel_specialty_data_0 = load_datasets(file_path)
    
    # Process the data with parallel execution
    _ = get_query_positives(
        icd_reference_lookup=icd_reference_lookup, 
        multilabel_specialty_data=multilabel_specialty_data_0,
        chunk_index=args.chunk,
        total_chunks=args.total_chunks,
        use_parallelism=not args.no_parallel,
        batch_size=args.batch_size,
        max_workers=args.workers
    )
```

## Key Improvements

1. **Command-line Argument Parsing**: Added CLI arguments to easily configure chunk processing, batch size, and parallel workers.

2. **Chunking Strategy**: Implemented intelligent chunking based on specialties rather than arbitrarily splitting the dictionary. This ensures each specialty is fully processed by one worker.

3. **Parallelization**: Added ThreadPoolExecutor to process multiple queries concurrently while respecting batch size limits.

4. **Progress Tracking**: Used tqdm for visual progress tracking at both specialty and query levels.

5. **Error Handling**: Improved exception handling to ensure errors with individual queries don't crash the entire process.

6. **Intermediate Saving**: Each specialty's results are saved as soon as they're completed, ensuring no data loss if the process is interrupted.

7. **Skip Logic**: Added logic to skip already processed specialties, making it easy to resume after interruptions.

## Usage Examples

1. Process all specialties sequentially (no chunking):
```bash
python gpt_icd_code_generator.py
```

2. Process chunk 0 of 4 chunks with parallel execution:
```bash
python gpt_icd_code_generator.py --chunk 0 --total_chunks 4 --batch_size 50 --workers 8
```

3. Disable parallel processing if needed:
```bash
python gpt_icd_code_generator.py --chunk 0 --total_chunks 4 --no_parallel
```

This implementation maintains your existing code structure but adds significant efficiency improvements. The parallel processing is bounded by batch size to avoid overwhelming the API, and each chunk now represents a group of specialties rather than an arbitrary split.

Would you like me to explain any specific part of the implementation in more detail?​​​​​​​​​​​​​​​​
