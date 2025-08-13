# Use the following command to run the script
# N is the number of GPUs on your VM
# torchrun --nproc_per_node=2 bf16_sentence_embeddding_finetuning_icd_original_h100_optimized.py
# torchrun --nproc_per_node=2 bf16_sentence_embeddding_finetuning_icd_original_h100_optimized.py >torchrun_output_v45.log 2>&1  

import os
import sys
import math
import json
import time
import psutil
import torch
import pickle    
import signal
import random
import logging
import numpy as np
import transformers
import pandas as pd
from tqdm import tqdm
from torch import optim
from collections import Counter
import matplotlib.pyplot as plt

from datasets import Dataset
from datetime import datetime
from types import SimpleNamespace
from transformers import AutoConfig
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import EarlyStoppingCallback, TrainerCallback
from transformers import get_scheduler
from sentence_transformers.readers import InputExample
from sentence_transformers.evaluation import TripletEvaluator
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.datasets import SentenceLabelDataset
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction
from sentence_transformers.losses import TripletLoss, MultipleNegativesRankingLoss, GISTEmbedLoss
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, util, SentenceTransformerTrainingArguments, SentenceTransformerTrainer

# ************************* OPTIMIZED CODE *************************
DATASET_SIZE = 0
DATASET_MEMORY_SIZE_GB = 0

# Reduce memory fragmentation and improve caching
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

def set_optimal_environment():
    """Set environment variables for optimal performance with large datasets"""
    os.environ["TOKENIZERS_PARALLELISM"] = "true"  # Enable parallel tokenization
    os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"  # Reduce I/O overhead
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # Enable async CUDA operations
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"  # Reduce fragmentation
    
    # Optimize for large datasets
    os.environ["HF_DATASETS_CACHE"] = "/tmp/hf_cache"  # Use fast storage for cache
    os.environ["TRANSFORMERS_CACHE"] = "/tmp/transformers_cache"

def get_optimal_workers_count() -> int:
    """Calculate optimal workers based on dataset size and hardware"""
    cpu_cores = os.cpu_count()
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        # For very large datasets, use more workers
        optimal_workers = min(gpu_count * 6, cpu_cores - 2)  # Increased from 3 to 6
    else:
        optimal_workers = min(12, cpu_cores - 2)  # Increased worker count
    
    print(f'Recommended dataloader_num_workers : {optimal_workers}')    
    return optimal_workers

def get_pin_memory_flag(config):
    """Pin memory calculation for large datasets"""
    AVAILABLE_MEMORY_GB = psutil.virtual_memory().available / (1024 ** 3)
    
    # Estimate memory usage more accurately for large datasets
    try:
        file_size_gb = os.path.getsize(config.TRAIN_DATASET_PATH) / (1024 ** 3)
    except:
        file_size_gb = 10  # Default estimate
    
    # For datasets this large, we need significant headroom
    pin_memory = AVAILABLE_MEMORY_GB > file_size_gb * 1.5  # Reduced from 2x to 1.5x
    print(f'AVAILABLE_MEMORY_GB : {AVAILABLE_MEMORY_GB:.2f} | DATASET_SIZE_GB : {file_size_gb:.2f} | pin_memory : {pin_memory}')
    
    return pin_memory

# ************************* OPTIMIZED CODE *************************

def load_training_parameters():
    file_path = '../../datasets/dataset_training'
    
    config = SimpleNamespace(
        TRAIN_DATASET_PATH = f'{file_path}/triplet_dataset_v50_250_queries_10positives_50hn_train_08112025.csv',
        EVAL_DATASET_PATH = f'{file_path}/triplet_dataset_v50_250_queries_10positives_50hn_eval_08112025.csv',
        TEST_DATASET_PATH = f'{file_path}/triplet_dataset_v50_250_queries_10positives_50hn_test_08112025.csv',
        SAMPLE_EVALUATION_QUERIES = True,
        VERSION = 50,
        MODEL_NAME = "e5-large-v2",
        ATTENTION_PROBS_DROPOUT_PROB = 0.05,
        HIDDEN_DROPOUT_PROB = 0.05,
        LOSS_FUNCTION = "MNR",
        MODEL_TYPE = "finetuned",
        DATASET_TYPE = "icd",
        OUTPUT_DIR = "",
        DO_TRAIN = True,
        DO_EVAL = True,
        SEED = 412,
        LR = 2e-05,
        EPOCHS = 10,
        TRAIN_BATCH_SIZE = 512,  # This will be adjusted based on sequence length variance
        EVAL_BATCH_SIZE = 32,
        SCALING_FACTOR = 1,
        TRAIN_STEPS = 0,
        WEIGHT_DECAY = 0.01,
        WARMUP_STEPS = 500,
        WARMUP_RATIO = 0.1,
        FP_16 = False,
        BF_16 = True,
        BATCH_SAMPLER = BatchSamplers.NO_DUPLICATES,
        EVALUATION_STRATEGY = "steps",
        EVAL_STEPS = 2000,
        SAVE_STRATEGY = "steps",
        SAVE_STEPS = 2000,
        EVAL_DELAY = 2000,
        SAVE_TOTAL_LIMIT = 2,
        RUN_NAME = "turing",
        LOGGING_DIR = ".",
        LOGGING_STRATEGY = "steps",
        LOGGING_FIRST_STEP = True,
        LOGGING_STEPS = 100,  # Reduced from 50 to 100 to reduce I/O overhead
        LEARNING_RATE_SCHEDULER = "cosine",
        REPORT_TO = "mlflow",
        DISABLE_TQDM = False,
        LOAD_BEST_MODEL_AT_END = True,
        METRIC_FOR_BEST_MODEL = 'eval_loss',
        GREATER_IS_BETTER = False,
        EARLY_STOPPING_PATIENCE = 5,
        DATALOADER_DROP_LAST = True,
        DATALOADER_NUM_WORKERS = 0,  # Will be reset
        DATALOADER_PERSISTENT_WORKERS = True,
        PIN_MEMORY = False,  # Will be reset
        GRADIENT_ACCUMULATION_STEPS = 8,
        MAX_GRAD_NORM = 0.5,
        # ************************* NEW OPTIMIZED PARAMETERS *************************
        MAX_SEQ_LENGTH = 256,  # Reduced from potential 350+ to reduce padding waste
        PREFETCH_FACTOR = 4,  # Add prefetching for better pipeline
        SMART_BATCHING = True,  # Enable length-aware batching
        REDUCE_EVAL_SAMPLES = True,  # Further reduce eval samples for speed
    )
    
    return config

def load_dataset_pandas_optimized(config) -> pd.DataFrame:
    """Optimized dataset loading for very large datasets"""
    global DATASET_SIZE
    
    print("Loading datasets with optimizations...")
    
    # Use chunked reading for memory efficiency
    chunk_size = 100000  # Process 100k rows at a time
    
    def load_and_process_chunks(file_path, sample_size=None):
        chunks = []
        total_rows = 0
        
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            # Remove first column (index) and process
            chunk = chunk.iloc[:, 1:]
            
            # Remove na samples immediately
            chunk = chunk.dropna()
            
            # Use only needed columns and drop duplicates
            if 'anchor' in chunk.columns:
                chunk = chunk[['anchor', 'positives', 'negatives']].drop_duplicates()
            else:
                # Handle different column names if needed
                chunk = chunk.iloc[:, [1, 2, 3]].drop_duplicates()
                chunk.columns = ['anchor', 'positives', 'negatives']
            
            chunks.append(chunk)
            total_rows += len(chunk)
            
            # Early termination if we have enough samples
            if sample_size and total_rows >= sample_size:
                break
        
        if chunks:
            result = pd.concat(chunks, ignore_index=True)
            if sample_size:
                result = result.head(sample_size)
            return result
        return pd.DataFrame()
    
    # Load training data (full dataset)
    print("Loading training data...")
    train_data = load_and_process_chunks(config.TRAIN_DATASET_PATH)
    
    # Load eval data with sampling
    print("Loading evaluation data...")
    if config.SAMPLE_EVALUATION_QUERIES and config.REDUCE_EVAL_SAMPLES:
        # Further reduce eval samples for faster evaluation
        eval_data = load_and_process_chunks(config.EVAL_DATASET_PATH, sample_size=5000)
        
        if config.SAMPLE_EVALUATION_QUERIES and len(eval_data) > 0:
            if 'specialty' in eval_data.columns:
                specialties = eval_data['specialty'].unique()
                samples_per_specialty = 5  # Reduced from 10 to 5
                sampled_data = []
                for specialty in specialties:
                    specialty_data = eval_data[eval_data['specialty'] == specialty]
                    n_samples = min(samples_per_specialty, len(specialty_data))
                    if n_samples > 0:
                        sampled = specialty_data.sample(n=n_samples, random_state=412)
                        sampled_data.append(sampled)
                if sampled_data:
                    eval_data = pd.concat(sampled_data, ignore_index=True)
    else:
        eval_data = load_and_process_chunks(config.EVAL_DATASET_PATH, sample_size=10000)
    
    # Load test data with sampling  
    print("Loading test data...")
    test_data = load_and_process_chunks(config.TEST_DATASET_PATH, sample_size=10000)
    
    # Final cleanup and reset indices
    train_data.reset_index(drop=True, inplace=True)
    eval_data.reset_index(drop=True, inplace=True) 
    test_data.reset_index(drop=True, inplace=True)
    
    print(f"Final Training Samples: {train_data.shape}")
    print(f"Final Evaluation Samples: {eval_data.shape}")
    print(f"Final Testing Samples: {test_data.shape}")

    DATASET_SIZE = train_data.shape[0]
    return train_data, eval_data, test_data

def build_huggingface_dataset_optimized(config):
    """Build datasets with memory optimization"""
    train_data, eval_data, test_data = load_dataset_pandas_optimized(config)
    
    # Convert to HuggingFace datasets without keeping pandas in memory
    print("Converting to HuggingFace datasets...")
    train_dataset = Dataset.from_pandas(train_data, preserve_index=False)
    eval_dataset = Dataset.from_pandas(eval_data, preserve_index=False) 
    test_dataset = Dataset.from_pandas(test_data, preserve_index=False)
    
    # Clear pandas dataframes from memory
    del train_data, eval_data, test_data
    
    return train_dataset, eval_dataset, test_dataset

def load_model_loss(config):
    print(f"Loading Model {config.MODEL_NAME}")
    
    if config.MODEL_NAME == "pubmedbert":
        model_path = "../model/NeuML_pubmedbert-base-embeddings/"
        sentence_transformer_config = {
            "attention_probs_dropout_prob": config.ATTENTION_PROBS_DROPOUT_PROB, 
            "hidden_dropout_prob": config.ATTENTION_PROBS_DROPOUT_PROB,
            "max_seq_length": config.MAX_SEQ_LENGTH  # Add max sequence length
        }
        model = SentenceTransformer(model_path, config_kwargs=sentence_transformer_config)
        
        print(f"{config.MODEL_NAME} Model Initialized")
        print("...........")
        print(f"Initializing Loss : {config.LOSS_FUNCTION}")
        
        if config.LOSS_FUNCTION == "Triplet":
            loss = TripletLoss(model)
        elif config.LOSS_FUNCTION == "MNR":
            loss = MultipleNegativesRankingLoss(model)
        elif config.LOSS_FUNCTION == "GIST":
            guide_model_path = '../../model/bge-large-en-v1.5/'
            guide_model = SentenceTransformer(guide_model_path)
            loss = GISTEmbedLoss(model, guide_model)
        
        print("Loss Initialized")
        
    elif config.MODEL_NAME == "bge":
        model_path = "../model/bge-large-en-v1.5/"
        sentence_transformer_config = {
            "attention_probs_dropout_prob": config.ATTENTION_PROBS_DROPOUT_PROB, 
            "hidden_dropout_prob": config.ATTENTION_PROBS_DROPOUT_PROB,
            "max_seq_length": config.MAX_SEQ_LENGTH
        }
        model = SentenceTransformer(model_path, config_kwargs=sentence_transformer_config)
        
        print(f"{config.MODEL_NAME} Model Initialized")
        print("...........")
        print(f"Initializing Loss : {config.LOSS_FUNCTION}")
        
        if config.LOSS_FUNCTION == "Triplet":
            loss = TripletLoss(model)
        elif config.LOSS_FUNCTION == "MNR":
            loss = MultipleNegativesRankingLoss(model)
        elif config.LOSS_FUNCTION == "GIST":
            guide_model_path = '../../../model/bge-large-en-v1.5/'
            guide_model = SentenceTransformer(guide_model_path)
            loss = GISTEmbedLoss(model, guide_model)
        
        print("Loss Initialized")
        
    elif config.MODEL_NAME == "e5-large-v2":
        model_path = '../../../shekhar_tanwar/ICD-ICD-Triplet/model/e5-large-v2-20250331143312-finetuned-icd-v30/'
        
        sentence_transformer_config = {
            "attention_probs_dropout_prob": config.ATTENTION_PROBS_DROPOUT_PROB, 
            "hidden_dropout_prob": config.ATTENTION_PROBS_DROPOUT_PROB,
            "max_seq_length": config.MAX_SEQ_LENGTH
        }
        model = SentenceTransformer(model_path, config_kwargs=sentence_transformer_config)
        
        # Explicitly set max_seq_length on transformer module
        for module in model.modules():
            if hasattr(module, 'max_seq_length'):
                module.max_seq_length = config.MAX_SEQ_LENGTH
        
        print(f"{config.MODEL_NAME} Model Initialized with max_seq_length={config.MAX_SEQ_LENGTH}")
        print("...........")
        print(f"Initializing Loss : {config.LOSS_FUNCTION}")
        
        if config.LOSS_FUNCTION == "Triplet":
            loss = TripletLoss(model)
        elif config.LOSS_FUNCTION == "MNR":
            loss = MultipleNegativesRankingLoss(model)
        elif config.LOSS_FUNCTION == "GIST":
            guide_model_path = '../../../model/bge-large-en-v1.5/'
            guide_model = SentenceTransformer(guide_model_path)
            loss = GISTEmbedLoss(model, guide_model)
        
        print("Loss Initialized")

    return model, loss

def get_timestamp():
    date = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    item = date.split(" ")
    item1 = "".join(item[0].split("-"))
    item2 = "".join(item[1].split(":"))
    date_time = item1 + item2
    return date_time

def create_length_sorted_dataloader(dataset, batch_size, num_workers, pin_memory, persistent_workers, drop_last=True):
    """Create a DataLoader with length-based sorting for smart batching"""
    from torch.utils.data import DataLoader
    from sentence_transformers.datasets import SentenceLabelDataset
    
    # Convert dataset to InputExample format if needed
    examples = []
    for item in dataset:
        if hasattr(item, 'texts'):
            examples.append(item)
        else:
            # Convert dict to InputExample
            examples.append(InputExample(texts=[item['anchor'], item['positives'], item['negatives']]))
    
    # Sort by combined length of all texts for better batching
    def get_text_length(example):
        if hasattr(example, 'texts'):
            return sum(len(text.split()) for text in example.texts)
        return 0
    
    examples.sort(key=get_text_length)
    
    # Create DataLoader with sorted examples
    dataloader = DataLoader(
        examples,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle since we want length-sorted batches
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=drop_last,
        collate_fn=None  # Will use model's default collate function
    )
    
    return dataloader

def get_SentenceTransformerTrainingArguments(config):
    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir=config.OUTPUT_DIR,
        # Optional training parameters:
        do_train=config.DO_TRAIN,
        do_eval=config.DO_EVAL,
        seed=config.SEED,
        learning_rate=config.LR,
        num_train_epochs=config.EPOCHS,
        per_device_train_batch_size=config.TRAIN_BATCH_SIZE * config.SCALING_FACTOR,
        per_device_eval_batch_size=config.EVAL_BATCH_SIZE,
        weight_decay=config.WEIGHT_DECAY,
        warmup_steps=config.WARMUP_STEPS,
        fp16=config.FP_16,
        bf16=config.BF_16,
        batch_sampler=config.BATCH_SAMPLER,
        # Optional tracking/debugging parameters:
        eval_strategy=config.EVALUATION_STRATEGY,
        eval_steps=config.EVAL_STEPS,
        save_strategy=config.SAVE_STRATEGY,
        save_steps=config.SAVE_STEPS,
        save_total_limit=config.SAVE_TOTAL_LIMIT,
        run_name=config.RUN_NAME,
        logging_dir=config.LOGGING_DIR,
        logging_strategy=config.LOGGING_STRATEGY,
        logging_first_step=config.LOGGING_FIRST_STEP,
        logging_steps=config.LOGGING_STEPS,
        report_to=config.REPORT_TO,
        disable_tqdm=config.DISABLE_TQDM,
        load_best_model_at_end=config.LOAD_BEST_MODEL_AT_END,
        metric_for_best_model=config.METRIC_FOR_BEST_MODEL,
        greater_is_better=config.GREATER_IS_BETTER,
        dataloader_drop_last=config.DATALOADER_DROP_LAST,
        eval_delay=config.EVAL_DELAY,
        # ************************* OPTIMIZED DATALOADER SETTINGS *************************
        dataloader_num_workers=config.DATALOADER_NUM_WORKERS,
        dataloader_persistent_workers=config.DATALOADER_PERSISTENT_WORKERS,
        dataloader_pin_memory=config.PIN_MEMORY,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        max_grad_norm=config.MAX_GRAD_NORM,
        # ************************* NEW OPTIMIZED PARAMETERS *************************
        dataloader_prefetch_factor=config.PREFETCH_FACTOR if config.DATALOADER_NUM_WORKERS > 0 else None,
        # Enable gradient checkpointing for memory efficiency
        gradient_checkpointing=True,
        # Optimize for throughput
        tf32=True,  # Enable TF32 on Ampere GPUs for faster training
    )
    
    return args

class OptimizedProgressCallback(TrainerCallback):
    """Optimized progress callback with reduced I/O"""
    def __init__(self):
        self.last_log_time = time.time()
        self.log_interval = 30  # Log every 30 seconds instead of every 10 steps
    
    def on_step_end(self, args, state, control, **kwargs):
        current_time = time.time()
        if current_time - self.last_log_time >= self.log_interval:
            print(f"Step {state.global_step}/{state.max_steps}")
            self.last_log_time = current_time

def main(config, date_time):
    # ************************* OPTIMIZED ENVIRONMENT SETUP *************************
    set_optimal_environment()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Enable optimizations for H100
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    print("Loading Datasets with optimizations...")
    train_dataset, eval_dataset, test_dataset = build_huggingface_dataset_optimized(config)

    # ************************* OPTIMIZED DATALOADER SETTINGS *************************
    print("Setting Up Optimized Dataloader Settings")
    config.DATALOADER_NUM_WORKERS = get_optimal_workers_count()
    config.PIN_MEMORY = get_pin_memory_flag(config)
    
    # ************************* CORRECTED BATCH SIZE AND STEP CALCULATIONS *************************
    N = len(train_dataset)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    print(f'WORLD_SIZE : {world_size}')
    
    # Calculate effective batch size and steps correctly
    B_eff = (config.TRAIN_BATCH_SIZE * config.SCALING_FACTOR * world_size * config.GRADIENT_ACCUMULATION_STEPS)
    steps_per_epoch = math.ceil(N / B_eff)
    total_steps = steps_per_epoch * config.EPOCHS
    
    config.TRAIN_STEPS = total_steps
    config.WARMUP_STEPS = int(config.WARMUP_RATIO * total_steps)
    
    print(f"Dataset size: {N:,}")
    print(f"Effective batch size: {B_eff}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total training steps: {total_steps}")
    print(f"Warmup steps: {config.WARMUP_STEPS}")
    
    print("Datasets Loaded")
    print("..............")

    model, loss = load_model_loss(config)
    print("..............")
    
    print("Initializing Optimizer")
    # Use AdamW with optimized parameters for large-scale training
    optimizer = AdamW(
        model.parameters(), 
        lr=config.LR, 
        weight_decay=config.WEIGHT_DECAY,
        eps=1e-6,  # Slightly larger eps for stability
        betas=(0.9, 0.999)  # Standard betas
    )

    if config.LEARNING_RATE_SCHEDULER == "linear":
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer=optimizer, 
            num_warmup_steps=config.WARMUP_STEPS, 
            num_training_steps=config.TRAIN_STEPS
        )
    elif config.LEARNING_RATE_SCHEDULER == "cosine":
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer=optimizer, 
            num_warmup_steps=config.WARMUP_STEPS, 
            num_training_steps=config.TRAIN_STEPS,
            num_cycles=0.5,
            last_epoch=-1
        )
    elif config.LEARNING_RATE_SCHEDULER == "cosine_with_restarts":
        scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer=optimizer, 
            num_warmup_steps=config.WARMUP_STEPS, 
            num_training_steps=config.TRAIN_STEPS,
            num_cycles=1
        )
        
    print(f'Learning Rate Scheduler Selected : {config.LEARNING_RATE_SCHEDULER}')
    config.OUTPUT_DIR = f"../model/{config.MODEL_NAME}-{date_time}-{config.MODEL_TYPE}-icd-v{config.VERSION}/"
    config.RUN_NAME = f"{config.MODEL_NAME}_{config.MODEL_TYPE}_v{config.VERSION}"
    dev_evaluator_name = f"all_{config.MODEL_NAME}_dev"
    test_evaluator_name = f"all_{config.MODEL_NAME}_test"
    
    print("Load Sentence Transformer Training Arguments")
    args = get_SentenceTransformerTrainingArguments(config)
    
    dev_evaluator = TripletEvaluator(
        anchors=eval_dataset["anchor"],
        positives=eval_dataset["positives"],
        negatives=eval_dataset["negatives"],
        name=dev_evaluator_name,
        batch_size=config.EVAL_BATCH_SIZE * 2,  # Larger eval batch for efficiency
        show_progress_bar=False  # Reduce progress bar overhead
    )
    
    # ************************* SMART BATCHING IMPLEMENTATION *************************
    if config.SMART_BATCHING:
        print("Enabling smart batching for length-aware batch creation...")
        # Create custom data collator for smart batching
        smart_collate_fn = create_smart_batching_collate_fn(model, config.MAX_SEQ_LENGTH)
        
        trainer = SentenceTransformerTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            loss=loss,
            evaluator=dev_evaluator,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=config.EARLY_STOPPING_PATIENCE),
                OptimizedProgressCallback()  # Use optimized progress callback
            ],
            optimizers=(optimizer, scheduler),
            data_collator=model.smart_batching_collate,  # Enable smart batching
        )
    else:
        trainer = SentenceTransformerTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            loss=loss,
            evaluator=dev_evaluator,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=config.EARLY_STOPPING_PATIENCE),
                OptimizedProgressCallback()  # Use optimized progress callback
            ],
            optimizers=(optimizer, scheduler),
        )
    
    print("Training Started with optimizations")

    def signal_handler(sig, frame):
        print(f"Training Interrupted. Current Step : {trainer.state.global_step}")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    
    trainer.train()
    print("..............")
    print("Testing Model on Test Evaluator")
    
    test_evaluator = TripletEvaluator(
        anchors=test_dataset["anchor"],
        positives=test_dataset["positives"],
        negatives=test_dataset["negatives"],
        name=test_evaluator_name,
        batch_size=config.EVAL_BATCH_SIZE * 2,
        show_progress_bar=False
    )
    
    test_evaluator(model)
    print("..............")

    # Save the trained model
    print("Saving Best Model")
    model.save_pretrained(config.OUTPUT_DIR)

if __name__ == "__main__":
    print("Load Parameters")
    config = load_training_parameters()
    date_time = get_timestamp()
    main(config, date_time)