import os
import time
import json
import random
import logging
import openai
import pickle
import requests
import threading
import itertools
import warnings
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any, Set
from datetime import datetime
from collections import defaultdict

from azureml.core import Workspace
from azureml.core.authentication import ServicePrincipalAuthentication

# Fix for Pydantic imports
try:
    from pydantic import BaseModel, Field
except ImportError:
    try:
        from pydantic.v1 import BaseModel, Field
    except ImportError:
        from langchain_core.pydantic_v1 import BaseModel, Field

from langchain.chains import LLMChain
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain.output_parsers import PydanticOutputParser, JsonOutputKeyToolsParser, CommaSeparatedListOutputParser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("icd_generator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Filter warnings
warnings.filterwarnings("ignore")

class RateLimiter:
    """Token bucket rate limiter to prevent API rate limit errors"""
    def __init__(self, max_calls, time_period):
        self.max_calls = max_calls      # Maximum number of calls allowed in the time period
        self.time_period = time_period  # Time period in seconds
        self.calls = []                 # List of timestamps for recent calls
        self.lock = threading.Lock()    # Thread safety
        logger.info(f"Rate limiter initialized: {max_calls} calls per {time_period} seconds")
    
    def acquire(self):
        """Wait until a token is available before returning"""
        with self.lock:
            now = time.time()
            
            # Remove timestamps older than our time period
            self.calls = [t for t in self.calls if now - t < self.time_period]
            
            # If we're at capacity, wait until the oldest call expires
            if len(self.calls) >= self.max_calls:
                sleep_time = self.time_period - (now - self.calls[0])
                if sleep_time > 0:
                    logger.info(f"Rate limit reached. Waiting for {sleep_time:.2f} seconds")
                    time.sleep(sleep_time)
                    # Recursive call after waiting
                    return self.acquire()
            
            # Add the current timestamp to our list of calls
            self.calls.append(time.time())
            return True

class BearerAuth(requests.auth.AuthBase):
    def __init__(self, token):
        self.token = token
        
    def __call__(self, r):
        r.headers["Authorization"] = "Bearer " + self.token
        return r

def initialize_llm() -> AzureChatOpenAI:
    """Initialize the Azure OpenAI model with proper authentication"""
    logger.info("Initializing Azure OpenAI model")
    
    try:
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
        
        logger.info(f'Initializing Model: {os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]}')
        model = AzureChatOpenAI(
            deployment_name=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
            azure_endpoint=os.environ["AZURE_OPENAI_API_BASE"],
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
            max_tokens=4000,
            temperature=0.9,
            model_kwargs={"seed": 1337}
        )
        
        logger.info('Model Initialized Successfully')
        return model
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}", exc_info=True)
        raise

def get_query_icd_codes_v4(model: AzureChatOpenAI, target_specialty: str, medical_query: str, max_retries=3):
    """Get ICD codes for a medical query with retry logic"""
    # This version of the prompt can handle queries with location based information
    
    class SpecialiesResponse(BaseModel):
        queries: List[str] = Field(description="List of queries corresponding to user provided medical specialty")
    
    # Set up the output parser
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
    
    # Initialize retry counter
    retry_count = 0
    base_delay = 2
    
    while retry_count < max_retries:
        try:
            chain = LLMChain(prompt=prompt_template, llm=model, output_parser=output_parser)
            result = chain.invoke(inputs={
                "target_specialty": target_specialty, 
                "medical_query": medical_query, 
                "format_instructions": output_parser.get_format_instructions()
            })
            return result
            
        except Exception as e:
            retry_count += 1
            error_msg = str(e).lower()
            
            # Check if it's a rate limit error
            if any(term in error_msg for term in ["rate", "limit", "capacity", "throttl", "quota"]):
                wait_time = base_delay * (2 ** retry_count) + random.uniform(0, 1)
                logger.warning(f"Rate limit hit for '{medical_query}'. Retrying in {wait_time:.2f}s (Attempt {retry_count}/{max_retries})")
                time.sleep(wait_time)
            else:
                logger.error(f"Error processing query '{medical_query}': {str(e)}")
                if retry_count < max_retries:
                    wait_time = base_delay * (1.5 ** retry_count)
                    logger.info(f"Retrying in {wait_time:.2f}s (Attempt {retry_count}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Max retries exceeded for '{medical_query}'")
                    return "[]"  # Return empty list as fallback
    
    # If we've exhausted all retries
    logger.error(f"Failed to process query after {max_retries} attempts: {medical_query}")
    return "[]"

def load_datasets(file_path):
    """Load reference datasets and specialty data"""
    logger.info(f"Loading datasets from {file_path}")
    
    try:
        # Load ICD codes reference
        icd_reference = pd.read_csv('../../../dataset/icd10.csv').iloc[:,1:]
        icd_reference = icd_reference.iloc[:,13:15]
        icd_reference.columns = ['codes','description']
        icd_reference = icd_reference.drop_duplicates()
        
        # Create lookup dictionary
        icd_reference_lookup = {}
        for row in icd_reference.itertuples():
            icd_reference_lookup[row.codes] = row.description
        
        # Load specialty data from file
        with open(file_path, 'r') as file:
            multilabel_specialty_data = json.load(file)
        
        logger.info(f"Loaded {len(multilabel_specialty_data)} specialties and {len(icd_reference_lookup)} ICD codes")
        return icd_reference_lookup, multilabel_specialty_data
    
    except Exception as e:
        logger.error(f"Error loading datasets: {str(e)}", exc_info=True)
        raise

def process_with_queue(input_file="../gpt_triplet_data_v5/multilabel_specialty_data.json", 
                       chunk_index=0, total_chunks=1, calls_per_minute=30, min_delay=1.0):
    """
    Process medical queries using a persistent queue with rate limiting.
    
    Args:
        input_file: Path to the input JSON file with specialties and queries
        chunk_index: Which chunk of specialties to process (0-indexed)
        total_chunks: Total number of chunks to split specialties into
        calls_per_minute: Maximum API calls allowed per minute
        min_delay: Minimum delay between API calls in seconds
        
    Returns:
        Dictionary with processed results
    """
    # Setup directories
    output_dir = "../gpt_triplet_data_v5/gpt_based_positives_set_3/"
    queue_dir = f"{output_dir}queue/"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(queue_dir, exist_ok=True)
    
    # Define filenames for persistent storage
    queue_file = f"{queue_dir}query_queue_chunk_{chunk_index}.json"
    progress_file = f"{queue_dir}progress_chunk_{chunk_index}.json"
    results_dir = f"{output_dir}results/"
    os.makedirs(results_dir, exist_ok=True)
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"Starting processing of chunk {chunk_index} at {timestamp}")
    
    # Load reference data and specialty data
    if os.path.exists(f"{output_dir}splits/gpt_specialties_split_{chunk_index}.json"):
        logger.info(f"Loading split data from {output_dir}splits/gpt_specialties_split_{chunk_index}.json")
        split_file = f"{output_dir}splits/gpt_specialties_split_{chunk_index}.json"
        icd_reference_lookup, multilabel_specialty_data = load_datasets(split_file)
    else:
        logger.info(f"No split file found. Loading and splitting from main file: {input_file}")
        # Create splits directory
        splits_dir = f"{output_dir}splits/"
        os.makedirs(splits_dir, exist_ok=True)
        
        # Load full data
        with open(input_file, 'r') as f:
            full_data = json.load(f)
        
        # Get all specialties and split
        all_specialties = list(full_data.keys())
        chunk_size = len(all_specialties) // total_chunks + (1 if len(all_specialties) % total_chunks else 0)
        
        # Create splits
        for i in range(total_chunks):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, len(all_specialties))
            chunk_specialties = all_specialties[start_idx:end_idx]
            
            chunk_data = {specialty: full_data[specialty] for specialty in chunk_specialties}
            split_file = f"{splits_dir}gpt_specialties_split_{i}.json"
            
            with open(split_file, 'w') as f:
                json.dump(chunk_data, f, indent=2)
            
            logger.info(f"Created split {i} with {len(chunk_specialties)} specialties")
        
        # Load the split for this chunk
        split_file = f"{splits_dir}gpt_specialties_split_{chunk_index}.json"
        icd_reference_lookup, multilabel_specialty_data = load_datasets(split_file)
    
    # Initialize or load queue
    if os.path.exists(queue_file):
        logger.info(f"Loading existing queue from {queue_file}")
        with open(queue_file, 'r') as f:
            queue = json.load(f)
    else:
        logger.info("Creating new queue from specialty data")
        # Create queue from specialties in this chunk
        queue = []
        
        # Add all queries for assigned specialties to the queue
        for specialty, queries in multilabel_specialty_data.items():
            # Clean specialty name
            if '/' in specialty:
                clean_specialty = specialty.replace('/', '_')
            else:
                clean_specialty = specialty
                
            # Skip if this specialty already has a result file
            specialty_file = f"{results_dir}{clean_specialty}.json"
            if os.path.exists(specialty_file):
                logger.info(f"Skipping specialty '{specialty}' - already processed")
                continue
                
            # Add queries for this specialty to the queue
            if not queries:
                logger.warning(f"No queries found for specialty '{specialty}'")
                continue
                
            logger.info(f"Adding {len(queries)} queries for specialty '{specialty}' to queue")
            for query in queries:
                queue.append({
                    "specialty": clean_specialty,
                    "query": query,
                    "status": "pending",  # pending, processing, completed, failed
                    "attempts": 0,
                    "last_attempt": None,
                    "result": None
                })
        
        # Save initial queue
        with open(queue_file, 'w') as f:
            json.dump(queue, f, indent=2)
    
    # Load progress data if exists
    progress = {
        "total": len(queue),
        "completed": 0,
        "failed": 0,
        "start_time": timestamp,
        "last_update": timestamp,
        "specialties": {}
    }
    
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            progress = json.load(f)
    
    # Initialize model
    model = initialize_llm()
    
    # Initialize rate limiter
    rate_limiter = RateLimiter(max_calls=calls_per_minute, time_period=60)
    
    # Count pending items
    pending_count = sum(1 for item in queue if item["status"] == "pending")
    completed_count = sum(1 for item in queue if item["status"] == "completed")
    failed_count = sum(1 for item in queue if item["status"] == "failed")
    
    logger.info(f"Queue stats: {pending_count} pending, {completed_count} completed, {failed_count} failed")
    
    # Process queue with progress bar
    with tqdm(total=len(queue), initial=completed_count+failed_count) as pbar:
        pbar.set_description(f"Chunk {chunk_index}")
        
        # Dictionary to collect results by specialty
        results_by_specialty = defaultdict(dict)
        
        # Load any existing specialty results
        for specialty in set(item["specialty"] for item in queue):
            specialty_file = f"{results_dir}{specialty}.json"
            if os.path.exists(specialty_file):
                with open(specialty_file, 'r') as f:
                    results_by_specialty[specialty] = json.load(f)
                    
                    # Update progress for this specialty
                    if specialty not in progress["specialties"]:
                        progress["specialties"][specialty] = {
                            "total": sum(1 for item in queue if item["specialty"] == specialty),
                            "completed": len(results_by_specialty[specialty]),
                            "status": "in_progress"
                        }
        
        # Process each item in the queue
        for i, item in enumerate(queue):
            # Skip already completed or failed items
            if item["status"] in ["completed", "failed"]:
                continue
                
            specialty = item["specialty"]
            query = item["query"]
            
            # Check if specialty file already exists and contains this query
            specialty_file = f"{results_dir}{specialty}.json"
            if os.path.exists(specialty_file) and specialty in results_by_specialty and query in results_by_specialty[specialty]:
                item["status"] = "completed"
                item["result"] = results_by_specialty[specialty][query]
                pbar.update(1)
                continue
            
            # Mark as processing
            item["status"] = "processing"
            item["attempts"] += 1
            item["last_attempt"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Save queue state
            with open(queue_file, 'w') as f:
                json.dump(queue, f, indent=2)
            
            try:
                # Apply rate limiting
                rate_limiter.acquire()
                
                # Enforce minimum delay between calls
                time.sleep(min_delay)
                
                # Process the query
                logger.info(f"Processing: {specialty} - {query}")
                result = get_query_icd_codes_v4(model=model, target_specialty=specialty, medical_query=query)
                
                # Parse response
                if isinstance(result, str) and result.strip() == "[]":
                    # Empty result
                    filtered_codes = []
                elif isinstance(result, str):
                    # String result
                    codes = [code.strip() for code in result.split(',')]
                    filtered_codes = [code for code in codes if code in icd_reference_lookup]
                elif isinstance(result, list):
                    # List result
                    filtered_codes = [code for code in result if code in icd_reference_lookup]
                else:
                    # Unknown format
                    logger.warning(f"Unexpected result format: {type(result)}")
                    filtered_codes = []
                
                # Store result
                item["status"] = "completed"
                item["result"] = filtered_codes
                results_by_specialty[specialty][query] = filtered_codes
                
                # Update progress
                progress["completed"] += 1
                progress["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                if specialty not in progress["specialties"]:
                    progress["specialties"][specialty] = {
                        "total": sum(1 for x in queue if x["specialty"] == specialty),
                        "completed": sum(1 for x in queue if x["specialty"] == specialty and x["status"] == "completed"),
                        "status": "in_progress"
                    }
                else:
                    progress["specialties"][specialty]["completed"] = sum(
                        1 for x in queue if x["specialty"] == specialty and x["status"] == "completed"
                    )
                
                # Check if specialty is complete
                if all(x["status"] in ["completed", "failed"] for x in queue if x["specialty"] == specialty):
                    progress["specialties"][specialty]["status"] = "completed"
                
                # Save specialty results
                with open(specialty_file, 'w') as f:
                    json.dump(results_by_specialty[specialty], f, indent=2)
                
            except Exception as e:
                logger.error(f"Error processing {query}: {str(e)}", exc_info=True)
                
                if item["attempts"] >= 3:
                    # Mark as failed after 3 attempts
                    item["status"] = "failed"
                    progress["failed"] += 1
                else:
                    # Reset to pending for retry
                    item["status"] = "pending"
                
                # Add longer delay after errors
                time.sleep(5)
            
            # Update progress bar
            pbar.update(1)
            
            # Save queue and progress regularly
            if i % 5 == 0 or item["status"] in ["completed", "failed"]:
                with open(queue_file, 'w') as f:
                    json.dump(queue, f, indent=2)
                with open(progress_file, 'w') as f:
                    json.dump(progress, f, indent=2)
    
    # Final save
    with open(queue_file, 'w') as f:
        json.dump(queue, f, indent=2)
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)
    
    logger.info(f"Completed processing chunk {chunk_index}")
    return results_by_specialty

def combine_results(output_dir="../gpt_triplet_data_v5/gpt_based_positives_set_3/", combined_file="combined_results.json"):
    """Combine results from all chunks into a single file"""
    results_dir = f"{output_dir}results/"
    combined_results = {}
    
    logger.info(f"Combining results from {results_dir}")
    
    # Get all result files
    result_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
    
    for filename in result_files:
        file_path = os.path.join(results_dir, filename)
        specialty = filename.replace('.json', '')
        
        with open(file_path, 'r') as f:
            specialty_results = json.load(f)
        
        combined_results[specialty] = specialty_results
    
    # Save combined results
    with open(os.path.join(output_dir, combined_file), 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    logger.info(f"Combined {len(result_files)} specialty results into {combined_file}")
    return combined_results

def split_input_data(input_file, output_dir, num_chunks):
    """Split input data into chunks for distributed processing"""
    logger.info(f"Splitting {input_file} into {num_chunks} chunks")
    
    # Create output directory
    splits_dir = f"{output_dir}splits/"
    os.makedirs(splits_dir, exist_ok=True)
    
    # Load input data
    with open(input_file, 'r') as f:
        specialty_data = json.load(f)
    
    # Get list of specialties
    specialties = list(specialty_data.keys())
    chunk_size = len(specialties) // num_chunks + (1 if len(specialties) % num_chunks else 0)
    
    # Create chunks
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, len(specialties))
        
        chunk_specialties = specialties[start_idx:end_idx]
        chunk_data = {specialty: specialty_data[specialty] for specialty in chunk_specialties}
        
        # Save chunk
        chunk_file = f"{splits_dir}gpt_specialties_split_{i}.json"
        with open(chunk_file, 'w') as f:
            json.dump(chunk_data, f, indent=2)
        
        logger.info(f"Chunk {i}: {len(chunk_specialties)} specialties, saved to {chunk_file}")
    
    return True

def main():
    """Main entry point with argument parsing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process medical specialties and generate ICD codes with rate limiting')
    
    # Define different operation modes as subparsers
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')
    
    # Split mode
    split_parser = subparsers.add_parser('split', help='Split input data into chunks')
    split_parser.add_argument('--input', type=str, default="../gpt_triplet_data_v5/multilabel_specialty_data.json",
                         help='Input file path')
    split_parser.add_argument('--output', type=str, default="../gpt_triplet_data_v5/gpt_based_positives_set_3/",
                         help='Output directory')
    split_parser.add_argument('--chunks', type=int, default=4,
                         help='Number of chunks to split into')
    
    # Process mode
    process_parser = subparsers.add_parser('process', help='Process a chunk of data')
    process_parser.add_argument('--input', type=str, default="../gpt_triplet_data_v5/multilabel_specialty_data.json",
                           help='Input file path')
    process_parser.add_argument('--output', type=str, default="../gpt_triplet_data_v5/gpt_based_positives_set_3/",
                           help='Output directory')
    process_parser.add_argument('--chunk', type=int, default=0,
                           help='Chunk index to process (0-indexed)')
    process_parser.add_argument('--chunks', type=int, default=4,
                           help='Total number of chunks')
    process_parser.add_argument('--rate', type=int, default=30,
                           help='Maximum API calls per minute')
    process_parser.add_argument('--delay', type=float, default=1.0,
                           help='Minimum delay between API calls in seconds')
    
    # Combine mode
    combine_parser = subparsers.add_parser('combine', help='Combine results from all chunks')
    combine_parser.add_argument('--output', type=str, default="../gpt_triplet_data_v5/gpt_based_positives_set_3/",
                           help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    if hasattr(args, 'output'):
        os.makedirs(args.output, exist_ok=True)
    
    if args.mode == 'split':
        split_input_data(args.input, args.output, args.chunks)
        
    elif args.mode == 'process':
        logger.info(f"Processing chunk {args.chunk} of {args.chunks} with rate limit {args.rate}/minute")
        process_with_queue(
            input_file=args.input,
            chunk_index=args.chunk,
            total_chunks=args.chunks,
            calls_per_minute=args.rate,
            min_delay=args.delay
        )
        
    elif args.mode == 'combine':
        combine_results(args.output)
    
    else:
        parser.print_help()
    
    return 0

if __name__ == "__main__":
    """
    Usage Examples:
    
    1. Split input data into chunks:
       python gpt_icd_code_generator.py split --input specialties_data.json --output ../output_dir/ --chunks 4
    
    2. Process a specific chunk with rate limiting:
       python gpt_icd_code_generator.py process --chunk 0 --chunks 4 --rate 15 --delay 2.0
    
    3. Combine results from all chunks:
       python gpt_icd_code_generator.py combine --output ../output_dir/
    
    4. Run all chunks sequentially (for overnight processing):
       for i in {0..3}; do python gpt_icd_code_generator.py process --chunk $i --chunks 4 --rate 20 --delay 3.0; done
    
    Notes:
    - Use --rate to control API calls per minute (lower for stricter rate limits)
    - Use --delay to add minimum time between API calls
    - The queue system allows stopping and resuming at any time
    - Results are saved by specialty, so completed specialties are never reprocessed
    - Progress tracking shows completion status for each specialty
    """
    main()
