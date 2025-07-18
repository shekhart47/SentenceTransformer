# Use the following command to run the script
# N is the number of GPUs on your VM
# torchrun --nproc_per_node=2 fp16_sentence_embeddding_finetuning_icd_original_h100.py
# torchrun --nproc_per_node=2 fp16_sentence_embeddding_finetuning_icd_original_h100.py >torchrun_output_v45.log 2>&1  

import os # new code
import sys # new code
import math
import json
import time
import psutil
import torch
import pickle    
import signal # new code
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

# ************************* new code *************************
DATASET_SIZE = 0
DATASET_MEMORY_SIZE_GB = 0


# get the optimal number of workers to load dataset faster
def get_optimal_workers_count() -> int:
    cpu_cores = os.cpu_count()
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        optimal_workers = min(gpu_count * 3, cpu_cores - 1)
    else:
        optimal_workers = min(6, cpu_cores - 1)
    
    print(f'Recommended dataloader_num_workers : {optimal_workers}')    

    return optimal_workers

# pin dataset into CPU memory for faster loading into GPUs
def get_pin_memory_flag(config):
    
    AVAILABLE_MEMORY_GB = psutil.virtual_memory().available / (1024 ** 3)
    
    DATASET_MEMORY_SIZE_GB = (os.path.getsize(config.TRAIN_DATASET_PATH) / (1024 ** 2)) / 1000
    pin_memory = AVAILABLE_MEMORY_GB > DATASET_MEMORY_SIZE_GB * 2 # need 2x headroom
    print(f'AVAILABLE_MEMORY_GB : {AVAILABLE_MEMORY_GB} | DATASET_MEMORY_SIZE_GB : {DATASET_MEMORY_SIZE_GB} pin_memory : {pin_memory}')
    
    return pin_memory


# ************************* new code *************************

# change the settings in this function
def load_training_parameters():
    file_path = '../../datasets/dataset_training'
    
    config = SimpleNamespace(
        TRAIN_DATASET_PATH = f'{file_path}/triplet_dataset_v45_240_queries_10positives_50hn_train.csv', # change the dataset type here
        EVAL_DATASET_PATH = f'{file_path}/triplet_dataset_v45_240_queries_10positives_50hn_eval.csv',
        TEST_DATASET_PATH = f'{file_path}/triplet_dataset_v45_240_queries_10positives_50hn_test.csv',
        SAMPLE_EVALUATION_QUERIES = True,
        VERSION = 45,
        MODEL_NAME = "qwen-0.6b", # Updated model name
        QWEN_MODEL_PATH = "../model/qwen-0.6b/", # Add local path for Qwen model
        ATTENTION_PROBS_DROPOUT_PROB = 0.05, #possible choices 0.2,0.3,0.5(higher values == lower train performance, may increase test performance)
        HIDDEN_DROPOUT_PROB = 0.05,
        LOSS_FUNCTION = "MNR", # possible choices "Triplet","MNR","GIST"
        MODEL_TYPE = "finetuned", # possible choices "finetuned","pretrained"
        DATASET_TYPE = "icd", # possible choices "icd","cpt" please make sure the DATASET_PATH and the DATASET_TYPE are compatible
        OUTPUT_DIR = "",
        DO_TRAIN = True,
        DO_EVAL = True,
        SEED = 412,
        LR = 4e-06, # default value 1e-5 for BGE, 2e-5 for pubmed_bert # LR is multipled by math.sqrt(SCALING_FACTOR)
        EPOCHS = 10,
        TRAIN_BATCH_SIZE = 32, #32 #256    # train batch size is multipled by SCALING_FACTOR
        EVAL_BATCH_SIZE = 32,
        SCALING_FACTOR = 1,
        TRAIN_STEPS = 0, # will be set in main
        WEIGHT_DECAY = 0.01, # original value 0.02 # L2 regularization paremeter
        WARMUP_STEPS = 500, # will be reset in main function
        FP_16 = False,
        BF_16 = True, #use on A100 / H100
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
        LOGGING_STEPS = 50,
        LEARNING_RATE_SCHEDULER = "cosine", # linear, cosine, cosine_with_restarts
        REPORT_TO = "mlflow",
        DISABLE_TQDM = False,
        LOAD_BEST_MODEL_AT_END = True,
        METRIC_FOR_BEST_MODEL  = 'eval_loss',
        GREATER_IS_BETTER = False,
        EARLY_STOPPING_PATIENCE = 5,
        DATALOADER_DROP_LAST = True,
        # pin dataset into CPU memory for faster loading into GPUs
        DATALOADER_NUM_WORKERS =  0, # will be reset in the main function
        DATALOADER_PERSISTENT_WORKERS = True,
        PIN_MEMORY = False, # will be reset in the main function,
        GRADIENT_ACCUMULATION_STEPS = 8,
        MAX_GRAD_NORM = 0.5, 
        
    )
    
    return config

def create_anchor_specialty_column(df):
    """
    Create anchor_specialty column using the prompting scheme for Qwen instruction tuning
    """
    task_tmpl = "Given a medical query in the field of {spec}, retrieve the relevant ICD description"
    
    def create_anchor_text(row):
        # Handle the case where the dataset might have 'query' or 'anchor' column
        query_text = row.get('query', row.get('anchor', ''))
        specialty = row.get('specialty', '')
        
        anchor_text = f"Instruct: {task_tmpl.format(spec=specialty)}\nQuery: {query_text}"
        return anchor_text
    
    df['anchor_specialty'] = df.apply(create_anchor_text, axis=1)
    print(f"Created anchor_specialty column. Sample entry:")
    print(f"{df['anchor_specialty'].iloc[0]}")
    print("=" * 50)
    
    return df

def load_dataset_pandas(config) -> pd.DataFrame:

    global DATASET_SIZE
    
    train_data = pd.read_csv(config.TRAIN_DATASET_PATH).iloc[:,1:]
    eval_data = pd.read_csv(config.EVAL_DATASET_PATH).iloc[:,1:]
    test_data = pd.read_csv(config.TEST_DATASET_PATH).iloc[:,1:]

    # these datasets have the following columns : ['specialty', 'anchor', 'positives', 'negatives']
    # OR ['specialty', 'query', 'positives', 'negatives']
    
    if config.SAMPLE_EVALUATION_QUERIES:
    
        specialties = eval_data['specialty'].unique()
        samples_per_specialty = 10
        sampled_data = []
        for specialty in specialties:
            specialty_data = eval_data[eval_data['specialty'] == specialty]
            n_samples = min(samples_per_specialty, len(specialty_data))
            sampled = specialty_data.sample(n = n_samples, random_state = 412)
            sampled_data.append(sampled)
        eval_data = pd.concat(sampled_data, ignore_index = True)

        
    # remove na samples
    train_data = train_data.dropna()
    eval_data = eval_data.dropna()
    test_data = test_data.dropna()
    
    # Create anchor_specialty column for instruction tuning
    print("Creating anchor_specialty column for Qwen instruction tuning...")
    train_data = create_anchor_specialty_column(train_data)
    eval_data = create_anchor_specialty_column(eval_data)
    test_data = create_anchor_specialty_column(test_data)
    
    # Use anchor_specialty, positives, negatives columns and drop duplicates
    train_data = train_data[['anchor_specialty', 'positives', 'negatives']].drop_duplicates()
    eval_data = eval_data[['anchor_specialty', 'positives', 'negatives']].drop_duplicates()
    test_data = test_data[['anchor_specialty', 'positives', 'negatives']].drop_duplicates()
    
    # Rename anchor_specialty to anchor for compatibility with existing code
    train_data = train_data.rename(columns={'anchor_specialty': 'anchor'})
    eval_data = eval_data.rename(columns={'anchor_specialty': 'anchor'})
    test_data = test_data.rename(columns={'anchor_specialty': 'anchor'})
    
    # Reset indices once
    train_data.reset_index(drop=True, inplace=True)
    eval_data.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)
    
    # Log dataset sizes
    print(f"Total Training Samples: {train_data.shape}")
    print(f"Total Evaluation Samples: {eval_data.shape}")
    print(f"Total Testing Samples: {test_data.shape}")

    DATASET_SIZE = train_data.shape[0]

    return train_data, eval_data, test_data


def build_huggingface_dataset(config):
    
    train_data, eval_data, test_data = load_dataset_pandas(config)
    train_data = Dataset.from_pandas(pd.DataFrame(data=train_data))#, preserve_index = False).with_format("torch")
    eval_data = Dataset.from_pandas(pd.DataFrame(data=eval_data))#, preserve_index = False).with_format("torch")
    test_data = Dataset.from_pandas(pd.DataFrame(data=test_data))#, preserve_index = False).with_format("torch")
    
    return train_data, eval_data, test_data

def load_model_loss(config):
    print(f"Loading Model {config.MODEL_NAME}")
    
    if config.MODEL_NAME == "pubmedbert":
       
        #model_path = "../model/pubmedbert-20240820180156-finetuned-icd-v14/"
        #model_path = "../model/NeuML_pubmedbert-base-embeddings/"
        sentence_transformer_config = {"attention_probs_dropout_prob" : config.ATTENTION_PROBS_DROPOUT_PROB, "hidden_dropout_prob" : config.ATTENTION_PROBS_DROPOUT_PROB}
        model = SentenceTransformer(model_path, config_kwargs = sentence_transformer_config)
        #print(f"Model Configuration : {model[0].config}")
        
        print(f"{config.MODEL_NAME} Model Initialized")
        print("...........")
        print(f"Initalizing Loss : {config.LOSS_FUNCTION}")
        
        if config.LOSS_FUNCTION == "Triplet":
            loss = TripletLoss(model)
            
        if config.LOSS_FUNCTION == "MNR":
            loss = MultipleNegativesRankingLoss(model)
        
        if config.LOSS_FUNCTION == "GIST":
            guide_model_path = '../../model/bge-large-en-v1.5/'
            
            guide_model = SentenceTransformer(guide_model_path)
            loss = GISTEmbedLoss(model, guide_model)
        
        print("Loss Initialized")
        
        
    elif config.MODEL_NAME == "bge":
        
        model_path = "../model/bge-large-en-v1.5/"
        sentence_transformer_config = {"attention_probs_dropout_prob" : config.ATTENTION_PROBS_DROPOUT_PROB, "hidden_dropout_prob" : config.ATTENTION_PROBS_DROPOUT_PROB}
        model = SentenceTransformer(model_path, config_kwargs = sentence_transformer_config)
        #print(f"Model Configuration : {model[0].config}")
        print(f"{config.MODEL_NAME} Model Initialized")
        print("...........")
        print(f"Initalizing Loss : {config.LOSS_FUNCTION}")
        
        if config.LOSS_FUNCTION == "Triplet":
            loss = TripletLoss(model)
            
        if config.LOSS_FUNCTION == "MNR":
            loss = MultipleNegativesRankingLoss(model)
        
        if config.LOSS_FUNCTION == "GIST":
            guide_model_path = '../../../model/bge-large-en-v1.5/'
                
            guide_model = SentenceTransformer(guide_model_path)
            loss = GISTEmbedLoss(model, guide_model)
        
        print("Loss Initialized")
        
        
    elif config.MODEL_NAME == "e5-large-v2" :

        model_path = '../../../shekhar_tanwar/ICD-ICD-Triplet/model/e5-large-v2-20250331143312-finetuned-icd-v30/'
        #model_path = '../../../shekhar_tanwar/ICD-ICD-Triplet/model/e5-large-v2/'
        
        sentence_transformer_config = {"attention_probs_dropout_prob" : config.ATTENTION_PROBS_DROPOUT_PROB, "hidden_dropout_prob" : config.ATTENTION_PROBS_DROPOUT_PROB}
        model = SentenceTransformer(model_path, config_kwargs = sentence_transformer_config)
        #print(f"Model Configuration : {model[0].config}")
        print(f"{config.MODEL_NAME} Model Initialized")
        print("...........")
        print(f"Initalizing Loss : {config.LOSS_FUNCTION}")
        
        if config.LOSS_FUNCTION == "Triplet":
            loss = TripletLoss(model)
            
        if config.LOSS_FUNCTION == "MNR":
            loss = MultipleNegativesRankingLoss(model)
        
        if config.LOSS_FUNCTION == "GIST":
            guide_model_path = '../../../model/bge-large-en-v1.5/'
                
            guide_model = SentenceTransformer(guide_model_path)
            loss = GISTEmbedLoss(model, guide_model)
        
        print("Loss Initialized")
        
    elif config.MODEL_NAME == "qwen-0.6b":
        
        model_path = config.QWEN_MODEL_PATH
        sentence_transformer_config = {"attention_probs_dropout_prob" : config.ATTENTION_PROBS_DROPOUT_PROB, "hidden_dropout_prob" : config.ATTENTION_PROBS_DROPOUT_PROB}
        model = SentenceTransformer(model_path, config_kwargs = sentence_transformer_config)
        #print(f"Model Configuration : {model[0].config}")
        print(f"{config.MODEL_NAME} Model Initialized")
        print("...........")
        print(f"Initalizing Loss : {config.LOSS_FUNCTION}")
        
        if config.LOSS_FUNCTION == "Triplet":
            loss = TripletLoss(model)
            
        if config.LOSS_FUNCTION == "MNR":
            loss = MultipleNegativesRankingLoss(model)
        
        if config.LOSS_FUNCTION == "GIST":
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

def get_SentenceTransformerTrainingArguments(config):
    
    
    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir= config.OUTPUT_DIR,
        # Optional training parameters:
        do_train = config.DO_TRAIN,
        do_eval = config.DO_EVAL,
        seed = config.SEED,
        learning_rate = config.LR ,#* math.sqrt(config.SCALING_FACTOR),
        num_train_epochs= config.EPOCHS,
        per_device_train_batch_size= config.TRAIN_BATCH_SIZE * config.SCALING_FACTOR,
        per_device_eval_batch_size= config.EVAL_BATCH_SIZE,
        weight_decay = config.WEIGHT_DECAY,
        warmup_steps = config.WARMUP_STEPS,
        #batch_sampler = BatchSamplers.NO_DUPLICATES, #, for MNR Loss
        fp16=config.FP_16,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=config.BF_16,  # Set to True if you have a GPU that supports BF16, eg : A100 
        batch_sampler=config.BATCH_SAMPLER,  # losses that use "in-batch negatives" benefit from no duplicates
        # Optional tracking/debugging parameters:
        eval_strategy= config.EVALUATION_STRATEGY,
        eval_steps= config.EVAL_STEPS,
        save_strategy= config.SAVE_STRATEGY,
        save_steps= config.SAVE_STEPS,
        save_total_limit= config.SAVE_TOTAL_LIMIT,
        run_name= config.RUN_NAME,  # Will be used in W&B if `wandb` is installed
        logging_dir = config.LOGGING_DIR,
        logging_strategy =  config.LOGGING_STRATEGY,
        logging_first_step = config.LOGGING_FIRST_STEP,
        logging_steps = config.LOGGING_STEPS,
        report_to = config.REPORT_TO,
        disable_tqdm = config.DISABLE_TQDM,
        load_best_model_at_end = config.LOAD_BEST_MODEL_AT_END,
        metric_for_best_model = config.METRIC_FOR_BEST_MODEL,
        greater_is_better=config.GREATER_IS_BETTER,
        dataloader_drop_last = config.DATALOADER_DROP_LAST,
        eval_delay = config.EVAL_DELAY,
        # ************************* new code *************************
        dataloader_num_workers = config.DATALOADER_NUM_WORKERS,
        dataloader_persistent_workers = config.DATALOADER_PERSISTENT_WORKERS,
        dataloader_pin_memory = config.PIN_MEMORY,
        #gradient_accumulation_steps = config.GRADIENT_ACCUMULATION_STEPS,
        max_grad_norm = config.MAX_GRAD_NORM,

    )
    
    return args

def main(config, date_time):
    #Parameters

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Set environment variables for better multi-GPU performance
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    
    print("Loading Datasets")
    train_dataset, eval_dataset, test_dataset = build_huggingface_dataset(config)

    # ************************* new code *************************
    print("Setting Up Dataloader Num Workers & PIN Meory Flag")
    config.DATALOADER_NUM_WORKERS = get_optimal_workers_count()
    config.PIN_MEMORY = get_pin_memory_flag(config)
    # ************************* new code *************************

    length_training_data = len(train_dataset)
    
    config.WARMUP_STEPS = math.ceil((length_training_data) * config.EPOCHS * 0.1)
    config.TRAIN_STEPS = int(length_training_data/config.TRAIN_BATCH_SIZE)
    print("Datasets Loaded")
    print("..............")

    model, loss = load_model_loss(config)
    print("..............")
    
    print("Initializing Optimizer")
#    optimizer = transformers.AdamW(model.parameters(),lr=config.LR, weight_decay = config.WEIGHT_DECAY)
    optimizer = AdamW(model.parameters(), lr=config.LR, weight_decay = config.WEIGHT_DECAY)

    if config.LEARNING_RATE_SCHEDULER == "linear":
        
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer = optimizer, 
            num_warmup_steps = config.WARMUP_STEPS, 
            num_training_steps = config.TRAIN_STEPS)

    elif config.LEARNING_RATE_SCHEDULER == "cosine":
        
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer = optimizer, 
            num_warmup_steps = config.WARMUP_STEPS, 
            num_training_steps = config.TRAIN_STEPS,
            num_cycles = 0.5,
            last_epoch = -1)

    elif config.LEARNING_RATE_SCHEDULER == "cosine_with_restarts":

        scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer = optimizer, 
            num_warmup_steps = config.WARMUP_STEPS, 
            num_training_steps = config.TRAIN_STEPS,
            num_cycles = 1
        )
    
        
    print(f'Learning Rate Scheduler Selected : {config.LEARNING_RATE_SCHEDULER}')
    config.OUTPUT_DIR = f"../model/{config.MODEL_NAME}-{date_time}-{config.MODEL_TYPE}-icd-v{config.VERSION}/"
    config.RUN_NAME = f"{config.MODEL_NAME}_{config.MODEL_TYPE}_v{config.VERSION}"
    dev_evaluator_name = f"all_{config.MODEL_NAME}_dev"
    test_evaluator_name = f"all_{config.MODEL_NAME}_test"
    
    print("Load Sentence Transformer Training Arguments")
    args = get_SentenceTransformerTrainingArguments(config)
    
    # set warmup steps
    
    dev_evaluator = TripletEvaluator(
                    anchors=eval_dataset["anchor"],
                    positives=eval_dataset["positives"],
                    negatives=eval_dataset["negatives"],
                    name=dev_evaluator_name,)
    
    
    trainer = SentenceTransformerTrainer(
                model=model,
                args=args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                loss=loss,
                evaluator=dev_evaluator,
                callbacks = [EarlyStoppingCallback(early_stopping_patience = config.EARLY_STOPPING_PATIENCE)], # stop training if eval_loss does not improve for 3 epochs,
                optimizers = (optimizer , scheduler),
    )
    print("Training Started")

    # ************************* new code *************************
    
    def signal_handler(sig, frame):
        print(f"Training Interrupted. Current Step : {trainer.state.global_step}")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    class ProgressCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step % 10 == 0:
                print(f"Step {state.global_step}/{state.max_steps}")

    trainer.add_callback(ProgressCallback())                

    # ************************* new code *************************
    
    trainer.train()
    print("..............")
    print("Testing Model on Test Evaluator")
    
    test_evaluator = TripletEvaluator(
                    anchors=test_dataset["anchor"],
                    positives=test_dataset["positives"],
                    negatives=test_dataset["negatives"],
                    name=test_evaluator_name,)
    
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
