# Production-ready training script optimized for full dataset training on H100 GPUs
# Designed for high model quality with 3-4 day training timeline
# Run with: torchrun --nproc_per_node=2 optimized_sentence_embedding_finetuning.py

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
import gc
from datetime import datetime, timedelta

from datasets import Dataset
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

# Enable TF32 for H100 optimization (significant speedup with minimal accuracy loss)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True  # Enable cuDNN autotuner
torch.backends.cudnn.deterministic = False  # Allow non-deterministic ops for speed

# Disable debugging features for production
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)

# Global variables for monitoring
DATASET_SIZE = 0
DATASET_MEMORY_SIZE_GB = 0
TRAINING_START_TIME = None

def set_optimal_environment():
    """Set environment variables for optimal H100 performance"""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Optimize NCCL for H100 with NVLink
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    os.environ["NCCL_P2P_DISABLE"] = "0"  # Enable P2P for NVLink
    os.environ["NCCL_IB_DISABLE"] = "0"
    os.environ["NCCL_TREE_THRESHOLD"] = "0"  # Always use tree algorithm
    
    # Optimize memory allocation for large datasets
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"
    
    # Enable CUDA graphs for better kernel launch overhead
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

def get_optimal_workers_count() -> int:
    """Calculate optimal number of workers for data loading"""
    cpu_cores = os.cpu_count()
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        # For large datasets, use more workers but cap at reasonable limit
        optimal_workers = min(gpu_count * 6, cpu_cores - 2, 12)
    else:
        optimal_workers = min(8, cpu_cores - 1)
    
    print(f'Recommended dataloader_num_workers: {optimal_workers}')
    return optimal_workers

def get_pin_memory_flag(config):
    """Determine if pinned memory should be used"""
    AVAILABLE_MEMORY_GB = psutil.virtual_memory().available / (1024 ** 3)
    
    # Estimate dataset memory size
    if os.path.exists(config.TRAIN_DATASET_PATH):
        DATASET_MEMORY_SIZE_GB = (os.path.getsize(config.TRAIN_DATASET_PATH) / (1024 ** 3))
    else:
        DATASET_MEMORY_SIZE_GB = 15  # Conservative estimate for 50M samples
    
    # Need enough headroom for data loading
    pin_memory = AVAILABLE_MEMORY_GB > (DATASET_MEMORY_SIZE_GB * 2.5)
    print(f'AVAILABLE_MEMORY_GB: {AVAILABLE_MEMORY_GB:.2f} | DATASET_MEMORY_SIZE_GB: {DATASET_MEMORY_SIZE_GB:.2f} | pin_memory: {pin_memory}')
    
    return pin_memory

def load_training_parameters():
    """Load and configure training parameters optimized for quality and speed balance"""
    file_path = '../../datasets/dataset_training'
    
    config = SimpleNamespace(
        # Dataset paths
        TRAIN_DATASET_PATH = f'{file_path}/triplet_dataset_v50_250_queries_10positives_50hn_train_08112025.csv',
        EVAL_DATASET_PATH = f'{file_path}/triplet_dataset_v50_250_queries_10positives_50hn_eval_08112025.csv',
        TEST_DATASET_PATH = f'{file_path}/triplet_dataset_v50_250_queries_10positives_50hn_test_08112025.csv',
        
        # Dataset configuration
        SAMPLE_EVALUATION_QUERIES = True,
        SAMPLE_TRAIN_DATA = False,  # Full dataset training as requested
        TRAIN_SAMPLE_RATIO = 1.0,
        
        # Model configuration
        VERSION = 50,
        MODEL_NAME = "e5-large-v2",
        ATTENTION_PROBS_DROPOUT_PROB = 0.1,  # Balanced dropout for generalization
        HIDDEN_DROPOUT_PROB = 0.1,
        LOSS_FUNCTION = "MNR",
        MODEL_TYPE = "finetuned",
        DATASET_TYPE = "icd",
        OUTPUT_DIR = "",
        
        # Training flags
        DO_TRAIN = True,
        DO_EVAL = True,
        
        # Hyperparameters optimized for BEST QUALITY with 3-4 day timeline
        SEED = 412,
        LR = 1e-05,  # Optimal LR for better final performance
        MAX_LR = 3e-05,  # Peak LR for OneCycleLR scheduler
        EPOCHS = 2,  # 2 full passes for better convergence (fits in 3-4 days)
        TRAIN_BATCH_SIZE = 256,  # Balanced for quality (smaller batch = better generalization)
        EVAL_BATCH_SIZE = 768,  # Large eval batch for speed
        SCALING_FACTOR = 1,
        TRAIN_STEPS = 0,  # Will be calculated
        WEIGHT_DECAY = 0.01,
        WARMUP_STEPS = 1000,  # More warmup for large dataset
        WARMUP_RATIO = 0.1,  # 10% warmup for better stability
        
        # Mixed precision - BF16 is optimal for H100
        FP_16 = False,
        BF_16 = True,
        TF_32 = True,  # Enable TF32 for H100
        
        # Batch sampling
        BATCH_SAMPLER = BatchSamplers.NO_DUPLICATES,
        
        # Evaluation strategy - more frequent for better model selection
        EVALUATION_STRATEGY = "steps",
        EVAL_STEPS = 5000,  # Evaluate every 5k steps for quality monitoring
        SAVE_STRATEGY = "steps",
        SAVE_STEPS = 5000,  # Save more checkpoints
        EVAL_DELAY = 2500,  # Start evaluation earlier
        SAVE_TOTAL_LIMIT = 10,  # Keep more checkpoints for analysis
        
        # Logging
        RUN_NAME = "h100_production",
        LOGGING_DIR = "./logs",
        LOGGING_STRATEGY = "steps",
        LOGGING_FIRST_STEP = True,
        LOGGING_STEPS = 200,  # Less frequent logging
        REPORT_TO = "none",  # Disable external reporting for speed
        
        # Performance optimizations
        DISABLE_TQDM = False,
        LOAD_BEST_MODEL_AT_END = True,
        METRIC_FOR_BEST_MODEL = 'eval_loss',
        GREATER_IS_BETTER = False,
        EARLY_STOPPING_PATIENCE = 5,  # Keep original patience for quality
        
        # DataLoader optimizations
        DATALOADER_DROP_LAST = True,
        DATALOADER_NUM_WORKERS = 0,  # Will be set dynamically
        DATALOADER_PERSISTENT_WORKERS = True,
        DATALOADER_PREFETCH_FACTOR = 4,  # Increased prefetch
        PIN_MEMORY = False,  # Will be set dynamically
        
        # Gradient accumulation for effective larger batch size
        GRADIENT_ACCUMULATION_STEPS = 3,  # Effective batch = 256 * 2 GPUs * 3 = 1536
        MAX_GRAD_NORM = 1.0,
        
        # Advanced optimizations
        GRADIENT_CHECKPOINTING = False,  # H100 has enough memory
        COMPILE_MODEL = False,  # Disabled due to SentenceTransformers compatibility
        USE_ONECYCLE_LR = False,  # Use cosine with restarts for 2 epochs
        USE_MIXED_PRECISION_OPTIMIZER = True,  # Use mixed precision optimizer
        USE_LION_OPTIMIZER = False,  # Option for Lion optimizer (better for long training)
        USE_FLASH_ATTENTION = False,  # Flash attention (if supported by model)
    )
    
    return config

def load_dataset_pandas_optimized(config) -> pd.DataFrame:
    """Load and preprocess datasets with optimization for large-scale training"""
    global DATASET_SIZE
    
    print("Loading datasets (this may take a few minutes for 50M samples)...")
    
    # Define columns to use
    usecols = ['specialty', 'anchor', 'positives', 'negatives']
    
    # Load with optimized settings
    dtype_dict = {
        'specialty': 'category',  # Use category dtype for memory efficiency
        'anchor': 'string',
        'positives': 'string',
        'negatives': 'string'
    }
    
    # Load training data in chunks if very large
    train_data = pd.read_csv(config.TRAIN_DATASET_PATH, usecols=usecols, dtype=dtype_dict)
    eval_data = pd.read_csv(config.EVAL_DATASET_PATH, usecols=usecols, dtype=dtype_dict)
    test_data = pd.read_csv(config.TEST_DATASET_PATH, usecols=usecols, dtype=dtype_dict)
    
    print(f"Raw data loaded. Processing...")
    
    # Sample evaluation data for faster evaluation
    if config.SAMPLE_EVALUATION_QUERIES:
        specialties = eval_data['specialty'].unique()
        samples_per_specialty = 30  # Reasonable sample for evaluation
        sampled_data = []
        for specialty in specialties:
            specialty_data = eval_data[eval_data['specialty'] == specialty]
            n_samples = min(samples_per_specialty, len(specialty_data))
            sampled = specialty_data.sample(n=n_samples, random_state=config.SEED)
            sampled_data.append(sampled)
        eval_data = pd.concat(sampled_data, ignore_index=True)
        print(f"Sampled evaluation data to {len(eval_data)} examples")
    
    # Efficient data cleaning
    initial_train_size = len(train_data)
    train_data = train_data.dropna(subset=['anchor', 'positives', 'negatives'])
    eval_data = eval_data.dropna(subset=['anchor', 'positives', 'negatives'])
    test_data = test_data.dropna(subset=['anchor', 'positives', 'negatives'])
    
    print(f"Removed {initial_train_size - len(train_data)} NaN samples from training data")
    
    # Convert to string type and select columns
    cols_to_keep = ['anchor', 'positives', 'negatives']
    train_data = train_data[cols_to_keep].astype(str)
    eval_data = eval_data[cols_to_keep].astype(str)
    test_data = test_data[cols_to_keep].astype(str)
    
    # Remove duplicates
    train_data = train_data.drop_duplicates()
    eval_data = eval_data.drop_duplicates()
    test_data = test_data.drop_duplicates()
    
    # Reset indices
    train_data.reset_index(drop=True, inplace=True)
    eval_data.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)
    
    # Log dataset sizes
    print(f"Total Training Samples: {train_data.shape}")
    print(f"Total Evaluation Samples: {eval_data.shape}")
    print(f"Total Testing Samples: {test_data.shape}")
    
    DATASET_SIZE = train_data.shape[0]
    
    # Force garbage collection
    gc.collect()
    
    return train_data, eval_data, test_data

def build_huggingface_dataset(config):
    """Build HuggingFace datasets with optimizations"""
    train_data, eval_data, test_data = load_dataset_pandas_optimized(config)
    
    # Convert to HuggingFace Dataset
    print("Converting to HuggingFace datasets...")
    train_dataset = Dataset.from_pandas(train_data, preserve_index=False)
    eval_dataset = Dataset.from_pandas(eval_data, preserve_index=False)
    test_dataset = Dataset.from_pandas(test_data, preserve_index=False)
    
    # Keep as arrow format for better memory efficiency with large datasets
    # Only convert to torch format when needed
    
    print("Datasets ready for training")
    return train_dataset, eval_dataset, test_dataset

def load_model_loss_optimized(config):
    """Load model and loss with optimizations"""
    print(f"Loading Model {config.MODEL_NAME}")
    
    model_path = '../../../shekhar_tanwar/ICD-ICD-Triplet/model/e5-large-v2-20250331143312-finetuned-icd-v30/'
    
    # Configure model with dropout
    sentence_transformer_config = {
        "attention_probs_dropout_prob": config.ATTENTION_PROBS_DROPOUT_PROB,
        "hidden_dropout_prob": config.HIDDEN_DROPOUT_PROB,
    }
    
    # Load model
    model = SentenceTransformer(model_path, config_kwargs=sentence_transformer_config)
    
    # Move model to GPU before any optimization
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Compile model with torch.compile for PyTorch 2.0+
    # DISABLED due to CUDAGraph compatibility issues with SentenceTransformers
    if config.COMPILE_MODEL and hasattr(torch, 'compile'):
        print("Note: torch.compile() disabled due to CUDAGraph compatibility issues with SentenceTransformers")
        print("Training will proceed without model compilation for stability")
        # Alternative optimization: Set model to use better CUDA kernels
        if hasattr(torch.backends.cudnn, 'benchmark'):
            torch.backends.cudnn.benchmark = True
            print("Enabled cuDNN benchmarking for kernel optimization")
    
    print(f"{config.MODEL_NAME} Model Initialized")
    print("Initializing Loss: MultipleNegativesRankingLoss")
    
    # Initialize MNR loss with temperature scaling
    loss = MultipleNegativesRankingLoss(
        model, 
        scale=20.0,  # Temperature scaling for better convergence
        similarity_fct=util.cos_sim  # Explicit cosine similarity
    )
    
    print("Loss Initialized")
    
    return model, loss

def get_timestamp():
    """Generate timestamp for model naming"""
    date = datetime.today().strftime('%Y%m%d%H%M%S')
    return date

def get_optimized_training_args(config):
    """Get optimized training arguments for production training"""
    
    args = SentenceTransformerTrainingArguments(
        # Required parameter
        output_dir=config.OUTPUT_DIR,
        
        # Training parameters
        do_train=config.DO_TRAIN,
        do_eval=config.DO_EVAL,
        seed=config.SEED,
        learning_rate=config.LR,
        num_train_epochs=config.EPOCHS,
        per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.EVAL_BATCH_SIZE,
        weight_decay=config.WEIGHT_DECAY,
        warmup_steps=config.WARMUP_STEPS,
        
        # Mixed precision (BF16 for H100)
        fp16=config.FP_16,
        bf16=config.BF_16,
        tf32=config.TF_32,
        
        # Batch sampling for MNR loss
        batch_sampler=config.BATCH_SAMPLER,
        
        # Evaluation and saving strategy
        eval_strategy=config.EVALUATION_STRATEGY,
        eval_steps=config.EVAL_STEPS,
        save_strategy=config.SAVE_STRATEGY,
        save_steps=config.SAVE_STEPS,
        save_total_limit=config.SAVE_TOTAL_LIMIT,
        eval_delay=config.EVAL_DELAY,
        
        # Logging
        run_name=config.RUN_NAME,
        logging_dir=config.LOGGING_DIR,
        logging_strategy=config.LOGGING_STRATEGY,
        logging_first_step=config.LOGGING_FIRST_STEP,
        logging_steps=config.LOGGING_STEPS,
        report_to=config.REPORT_TO,
        disable_tqdm=config.DISABLE_TQDM,
        
        # Model selection
        load_best_model_at_end=config.LOAD_BEST_MODEL_AT_END,
        metric_for_best_model=config.METRIC_FOR_BEST_MODEL,
        greater_is_better=config.GREATER_IS_BETTER,
        
        # DataLoader optimizations
        dataloader_drop_last=config.DATALOADER_DROP_LAST,
        dataloader_num_workers=config.DATALOADER_NUM_WORKERS,
        dataloader_persistent_workers=config.DATALOADER_PERSISTENT_WORKERS,
        dataloader_pin_memory=config.PIN_MEMORY,
        dataloader_prefetch_factor=config.DATALOADER_PREFETCH_FACTOR,
        
        # Gradient accumulation
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        max_grad_norm=config.MAX_GRAD_NORM,
        
        # Performance optimizations
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=25,  # Optimize DDP communication
        
        # Additional optimizations for large-scale training
        group_by_length=False,  # Not applicable for triplet data
        length_column_name="length",
        ddp_broadcast_buffers=False,
    )
    
    return args

class TrainingMonitorCallback(TrainerCallback):
    """Comprehensive training monitor with ETA and performance tracking"""
    
    def __init__(self, total_steps, log_interval=100):
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.start_time = time.time()
        self.step_times = []
        self.gpu_utils = []
        
    def on_step_end(self, args, state, control, **kwargs):
        current_step = state.global_step
        
        if current_step % self.log_interval == 0 and current_step > 0:
            # Calculate timing statistics
            elapsed_time = time.time() - self.start_time
            steps_per_second = current_step / elapsed_time
            remaining_steps = self.total_steps - current_step
            eta_seconds = remaining_steps / steps_per_second if steps_per_second > 0 else 0
            
            # Format ETA
            eta = str(timedelta(seconds=int(eta_seconds)))
            
            # GPU monitoring
            if torch.cuda.is_available():
                allocated_gb = torch.cuda.memory_allocated() / 1024**3
                reserved_gb = torch.cuda.memory_reserved() / 1024**3
                
                # Get GPU utilization
                try:
                    import nvidia_ml_py3 as nvml
                    nvml.nvmlInit()
                    handle = nvml.nvmlDeviceGetHandleByIndex(0)
                    util = nvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_util = util.gpu
                    self.gpu_utils.append(gpu_util)
                    avg_gpu_util = np.mean(self.gpu_utils[-100:])  # Rolling average
                except:
                    gpu_util = "N/A"
                    avg_gpu_util = "N/A"
                
                print(f"\n{'='*80}")
                print(f"Step: {current_step}/{self.total_steps} ({current_step/self.total_steps*100:.1f}%)")
                print(f"Speed: {steps_per_second:.2f} steps/sec | ETA: {eta}")
                print(f"GPU Memory: {allocated_gb:.2f}GB allocated, {reserved_gb:.2f}GB reserved")
                print(f"GPU Utilization: Current={gpu_util}%, Average={avg_gpu_util:.1f}%" if gpu_util != "N/A" else "")
                print(f"{'='*80}\n")
            
            # Clear cache periodically to prevent fragmentation
            if current_step % 1000 == 0:
                torch.cuda.empty_cache()
                gc.collect()

class AdaptiveSchedulerCallback(TrainerCallback):
    """Advanced learning rate scheduling with warmup restart"""
    
    def __init__(self, optimizer, num_training_steps, num_warmup_steps):
        self.optimizer = optimizer
        self.num_training_steps = num_training_steps
        self.num_warmup_steps = num_warmup_steps
        self.current_step = 0
        
    def on_step_begin(self, args, state, control, **kwargs):
        self.current_step = state.global_step
        
        # Implement OneCycle-like schedule
        if self.current_step < self.num_warmup_steps:
            # Warmup phase
            lr_scale = self.current_step / self.num_warmup_steps
        else:
            # Cosine annealing phase
            progress = (self.current_step - self.num_warmup_steps) / (self.num_training_steps - self.num_warmup_steps)
            lr_scale = 0.5 * (1 + math.cos(math.pi * progress))
        
        # Apply learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = args.learning_rate * lr_scale

class CheckpointCallback(TrainerCallback):
    """Smart checkpointing with best model tracking"""
    
    def __init__(self, output_dir, save_steps=10000):
        self.output_dir = output_dir
        self.save_steps = save_steps
        self.best_loss = float('inf')
        
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and 'eval_loss' in metrics:
            current_loss = metrics['eval_loss']
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                # Save best model
                best_model_path = os.path.join(self.output_dir, "best_model")
                kwargs['model'].save_pretrained(best_model_path)
                print(f"New best model saved with eval_loss: {current_loss:.4f}")

def estimate_training_time(dataset_size, batch_size, world_size, gradient_accumulation_steps):
    """Estimate total training time based on dataset and hardware"""
    effective_batch_size = batch_size * world_size * gradient_accumulation_steps
    steps_per_epoch = math.ceil(dataset_size / effective_batch_size)
    
    # Empirical estimates for H100 (adjust based on actual measurements)
    seconds_per_step = 0.5  # Conservative estimate for large batches
    total_seconds = steps_per_epoch * seconds_per_step
    
    return steps_per_epoch, timedelta(seconds=int(total_seconds))

def main(config, date_time):
    """Main training function optimized for production training"""
    global TRAINING_START_TIME
    TRAINING_START_TIME = time.time()
    
    # Set optimal environment
    set_optimal_environment()
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.95)
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    print("=" * 80)
    print("PRODUCTION TRAINING ON H100 GPUs")
    print("Target: High quality model in 3-4 days")
    print("=" * 80)
    
    # Load datasets
    print("\n📊 Loading Full Dataset (50M+ samples)...")
    train_dataset, eval_dataset, test_dataset = build_huggingface_dataset(config)
    
    # Configure DataLoader settings
    print("\n⚙️ Configuring Optimized DataLoader...")
    config.DATALOADER_NUM_WORKERS = get_optimal_workers_count()
    config.PIN_MEMORY = get_pin_memory_flag(config)
    
    # Calculate training steps
    N = len(train_dataset)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    print(f'\n🌍 World Size (GPUs): {world_size}')
    
    B_eff = config.TRAIN_BATCH_SIZE * world_size * config.GRADIENT_ACCUMULATION_STEPS
    steps_per_epoch = math.ceil(N / B_eff)
    total_steps = steps_per_epoch * config.EPOCHS
    
    config.TRAIN_STEPS = total_steps
    config.WARMUP_STEPS = int(config.WARMUP_RATIO * total_steps)
    
    # Estimate training time
    _, estimated_time_per_epoch = estimate_training_time(N, config.TRAIN_BATCH_SIZE, world_size, config.GRADIENT_ACCUMULATION_STEPS)
    
    print(f"\n📈 Training Configuration:")
    print(f"  - Dataset Size: {N:,} samples")
    print(f"  - Batch Size per GPU: {config.TRAIN_BATCH_SIZE}")
    print(f"  - Effective Batch Size: {B_eff:,}")
    print(f"  - Steps per Epoch: {steps_per_epoch:,}")
    print(f"  - Total Training Steps: {total_steps:,}")
    print(f"  - Warmup Steps: {config.WARMUP_STEPS:,}")
    print(f"  - Estimated Time per Epoch: {estimated_time_per_epoch}")
    print(f"  - Total Estimated Time: {estimated_time_per_epoch * config.EPOCHS}")
    
    # Load model and loss
    print("\n🤖 Loading Model and Loss Function...")
    model, loss = load_model_loss_optimized(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - Total Parameters: {total_params:,}")
    print(f"  - Trainable Parameters: {trainable_params:,}")
    
    # Initialize optimizer
    print("\n🔧 Initializing Optimizer...")
    
    if config.USE_MIXED_PRECISION_OPTIMIZER:
        # Use AdamW with mixed precision optimization
        optimizer = AdamW(
            model.parameters(),
            lr=config.LR,
            weight_decay=config.WEIGHT_DECAY,
            betas=(0.9, 0.999),
            eps=1e-6,
            fused=True if torch.cuda.is_available() else False  # Fused optimizer for H100
        )
    else:
        optimizer = AdamW(
            model.parameters(),
            lr=config.LR,
            weight_decay=config.WEIGHT_DECAY,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
    
    # Initialize learning rate scheduler
    if config.USE_ONECYCLE_LR:
        print("  - Using OneCycle Learning Rate Schedule")
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.MAX_LR,
            total_steps=config.TRAIN_STEPS,
            pct_start=config.WARMUP_RATIO,
            anneal_strategy='cos',
            cycle_momentum=False,
            div_factor=25.0,  # Initial lr = max_lr/25
            final_div_factor=10000.0,  # Final lr = max_lr/10000
        )
    else:
        print("  - Using Cosine with Restarts Schedule (optimal for 2 epochs)")
        scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=config.WARMUP_STEPS,
            num_training_steps=config.TRAIN_STEPS,
            num_cycles=config.EPOCHS,  # One restart per epoch
        )
    
    # Set output directory
    config.OUTPUT_DIR = f"../model/{config.MODEL_NAME}-{date_time}-{config.MODEL_TYPE}-icd-v{config.VERSION}-production/"
    config.RUN_NAME = f"{config.MODEL_NAME}_{config.MODEL_TYPE}_v{config.VERSION}_production"
    
    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(config.OUTPUT_DIR, "checkpoints"), exist_ok=True)
    
    # Save initial configuration
    config_dict = vars(config)
    with open(f"{config.OUTPUT_DIR}/training_config.json", 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)
    
    # Get training arguments
    print("\n📝 Configuring Training Arguments...")
    args = get_optimized_training_args(config)
    
    # Initialize evaluator
    print("\n📊 Setting up Evaluator...")
    dev_evaluator_name = f"eval_{config.MODEL_NAME}"
    dev_evaluator = TripletEvaluator(
        anchors=eval_dataset["anchor"],
        positives=eval_dataset["positives"],
        negatives=eval_dataset["negatives"],
        name=dev_evaluator_name,
        batch_size=config.EVAL_BATCH_SIZE,
        show_progress_bar=True,
    )
    
    # Initialize callbacks
    print("\n🔔 Setting up Callbacks...")
    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=config.EARLY_STOPPING_PATIENCE),
        TrainingMonitorCallback(total_steps=total_steps, log_interval=200),
        CheckpointCallback(output_dir=config.OUTPUT_DIR, save_steps=config.SAVE_STEPS),
    ]
    
    # Initialize trainer
    print("\n🚀 Initializing Trainer...")
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=dev_evaluator,
        callbacks=callbacks,
        optimizers=(optimizer, scheduler),
    )
    
    # Signal handler for graceful interruption
    def signal_handler(sig, frame):
        print(f"\n⚠️ Training Interrupted at Step {trainer.state.global_step}")
        checkpoint_path = f"{config.OUTPUT_DIR}/interrupted_checkpoint/"
        print(f"💾 Saving checkpoint to {checkpoint_path}")
        trainer.save_model(checkpoint_path)
        
        # Save training state
        state_path = f"{checkpoint_path}/training_state.json"
        state_info = {
            "interrupted_at_step": trainer.state.global_step,
            "total_steps": total_steps,
            "elapsed_time": time.time() - TRAINING_START_TIME,
            "timestamp": datetime.now().isoformat()
        }
        with open(state_path, 'w') as f:
            json.dump(state_info, f, indent=2)
        
        print("✅ Checkpoint saved successfully")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start training
    print("\n" + "=" * 80)
    print("🎯 STARTING TRAINING")
    print(f"📅 Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️ Estimated Completion: {(datetime.now() + estimated_time_per_epoch * config.EPOCHS).strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")
    
    # Training with error handling
    try:
        trainer.train()
        training_success = True
    except Exception as e:
        print(f"\n❌ Training Error: {e}")
        print("💾 Attempting to save current model state...")
        error_checkpoint_path = f"{config.OUTPUT_DIR}/error_checkpoint/"
        trainer.save_model(error_checkpoint_path)
        print(f"✅ Error checkpoint saved to {error_checkpoint_path}")
        training_success = False
        raise
    
    # Calculate actual training time
    training_time = time.time() - TRAINING_START_TIME
    
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETED" if training_success else "⚠️ TRAINING INTERRUPTED")
    print(f"⏱️ Total Training Time: {str(timedelta(seconds=int(training_time)))}")
    print(f"📊 Final Step: {trainer.state.global_step}/{total_steps}")
    print("=" * 80 + "\n")
    
    # Test evaluation
    print("📊 Running Final Evaluation on Test Set...")
    print("This may take a while with 4M+ test samples...")
    
    test_evaluator_name = f"test_{config.MODEL_NAME}"
    test_evaluator = TripletEvaluator(
        anchors=test_dataset["anchor"][:100000],  # Evaluate on subset for speed
        positives=test_dataset["positives"][:100000],
        negatives=test_dataset["negatives"][:100000],
        name=test_evaluator_name,
        batch_size=config.EVAL_BATCH_SIZE,
        show_progress_bar=True,
    )
    
    test_results = test_evaluator(model)
    print(f"\n📈 Test Results (100k sample): {test_results}")
    
    # Save the final model
    print("\n💾 Saving Final Model...")
    final_model_path = os.path.join(config.OUTPUT_DIR, "final_model")
    model.save_pretrained(final_model_path)
    
    # Save comprehensive training summary
    summary_path = os.path.join(config.OUTPUT_DIR, "training_summary.json")
    summary = {
        "model_name": config.MODEL_NAME,
        "version": config.VERSION,
        "dataset_size": N,
        "total_steps": total_steps,
        "completed_steps": trainer.state.global_step,
        "training_time_seconds": training_time,
        "training_time_human": str(timedelta(seconds=int(training_time))),
        "final_learning_rate": scheduler.get_last_lr()[0],
        "test_results": test_results,
        "effective_batch_size": B_eff,
        "num_gpus": world_size,
        "completion_timestamp": datetime.now().isoformat(),
        "model_parameters": {
            "total": total_params,
            "trainable": trainable_params
        },
        "hyperparameters": {
            "learning_rate": config.LR,
            "epochs": config.EPOCHS,
            "batch_size": config.TRAIN_BATCH_SIZE,
            "gradient_accumulation": config.GRADIENT_ACCUMULATION_STEPS,
            "warmup_steps": config.WARMUP_STEPS,
            "weight_decay": config.WEIGHT_DECAY,
            "dropout": config.ATTENTION_PROBS_DROPOUT_PROB
        }
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📁 Model and training summary saved to {config.OUTPUT_DIR}")
    
    # Performance recommendations
    print("\n" + "=" * 80)
    print("💡 PERFORMANCE ANALYSIS & RECOMMENDATIONS")
    print("=" * 80)
    
    avg_steps_per_second = trainer.state.global_step / training_time if training_time > 0 else 0
    samples_per_second = avg_steps_per_second * B_eff
    
    print(f"\n📊 Training Statistics:")
    print(f"  - Average Speed: {avg_steps_per_second:.2f} steps/sec")
    print(f"  - Throughput: {samples_per_second:.0f} samples/sec")
    print(f"  - GPU Efficiency: {samples_per_second / (world_size * 1000):.1f}k samples/sec/GPU")
    
    # Check if training was efficient
    if avg_steps_per_second < 1.0:
        print("\n⚠️ Training speed was slower than expected. Consider:")
        print("  1. Increasing batch size (current: {})".format(config.TRAIN_BATCH_SIZE))
        print("  2. Reducing gradient accumulation steps (current: {})".format(config.GRADIENT_ACCUMULATION_STEPS))
        print("  3. Checking CPU-GPU data transfer bottlenecks")
        print("  4. Monitoring with: watch -n 1 nvidia-smi")
    else:
        print("\n✅ Training speed was good!")
    
    print("\n🎯 Next Steps:")
    print("  1. Evaluate the model on your full test set if needed")
    print("  2. Run inference benchmarks to measure latency")
    print("  3. Consider quantization or distillation for deployment")
    print("  4. Monitor the saved checkpoints for best performance")
    
    print("\n" + "=" * 80)
    print("🎉 TRAINING PIPELINE COMPLETE!")
    print("=" * 80)

if __name__ == "__main__":
    print("🚀 Initializing Production Training Pipeline for H100 GPUs")
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔧 PyTorch Version: {torch.__version__}")
    print(f"🔧 Transformers Version: {transformers.__version__}")
    print(f"🔧 CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"🔧 CUDA Version: {torch.version.cuda}")
        print(f"🔧 cuDNN Version: {torch.backends.cudnn.version()}")
    
    print("\n" + "=" * 80)
    
    config = load_training_parameters()
    date_time = get_timestamp()
    
    try:
        main(config, date_time)
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("\n✅ Cleanup completed")