import argparse
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, AutoConfig
from transformers import AdamW, get_linear_schedule_with_warmup
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from typing import List, Dict, Union, Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt

class E5DistillationDataset(Dataset):
    """
    Dataset for distilling knowledge from a teacher E5 model to a student model
    """
    
    def __init__(
        self, 
        texts: List[str], 
        tokenizer, 
        teacher_embeddings: Optional[np.ndarray] = None,
        max_length: int = 512
    ):
        """
        Initialize the dataset.
        
        Args:
            texts: List of text samples
            tokenizer: Tokenizer for the model
            teacher_embeddings: Pre-computed teacher embeddings (optional)
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.teacher_embeddings = teacher_embeddings
        self.max_length = max_length
        
        # Format texts with appropriate E5 prefixes
        self.formatted_texts = []
        for text in texts:
            if not text.startswith(("query: ", "passage: ")):
                if len(text.strip().split()) <= 10:  # Heuristic for queries
                    self.formatted_texts.append(f"query: {text}")
                else:
                    self.formatted_texts.append(f"passage: {text}")
            else:
                self.formatted_texts.append(text)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.formatted_texts[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Convert to appropriate shape
        item = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'idx': idx  # Keep track of index for teacher embeddings
        }
        
        # Include teacher embeddings if available
        if self.teacher_embeddings is not None:
            item['teacher_embedding'] = torch.tensor(
                self.teacher_embeddings[idx], 
                dtype=torch.float32
            )
            
        return item

class E5StudentModel(nn.Module):
    """
    A smaller student model that mimics E5 embedding behavior
    """
    
    def __init__(
        self, 
        teacher_model_path: str, 
        hidden_size: int = 384,  # Reduced from typical 768
        hidden_layers: int = 6,  # Reduced from typical 12
        attention_heads: int = 6  # Reduced from typical 12
    ):
        """
        Initialize a student model based on teacher architecture but smaller.
        
        Args:
            teacher_model_path: Path to teacher model to copy configuration
            hidden_size: Size of hidden layers in student model
            hidden_layers: Number of transformer layers
            attention_heads: Number of attention heads
        """
        super(E5StudentModel, self).__init__()
        
        # Load teacher config to adapt
        teacher_config = AutoConfig.from_pretrained(teacher_model_path)
        
        # Create student config with reduced size
        student_config = AutoConfig.from_pretrained(
            teacher_model_path,
            hidden_size=hidden_size,
            num_hidden_layers=hidden_layers,
            num_attention_heads=attention_heads,
            intermediate_size=hidden_size * 4  # Typically 4x hidden size
        )
        
        # Initialize model with student config
        self.model = AutoModel.from_config(student_config)
        
        # Get embedding dimension from teacher
        self.teacher_dim = teacher_config.hidden_size
        
        # Projection layer to match teacher dimensions if different
        if hidden_size != self.teacher_dim:
            self.projection = nn.Linear(hidden_size, self.teacher_dim)
        else:
            self.projection = None
    
    def mean_pooling(self, token_embeddings, attention_mask):
        """Apply mean pooling to get sentence embedding"""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def forward(self, input_ids, attention_mask):
        """Forward pass through the model"""
        # Get token embeddings
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state
        
        # Apply mean pooling
        embeddings = self.mean_pooling(token_embeddings, attention_mask)
        
        # Project to teacher dimension if needed
        if self.projection is not None:
            embeddings = self.projection(embeddings)
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings

class E5Distiller:
    """
    Handles the knowledge distillation process from a teacher E5 model to a smaller student model
    """
    
    def __init__(
        self, 
        teacher_model_path: str,
        output_dir: str = "./distilled_model",
        hidden_size: int = 384,
        hidden_layers: int = 6,
        attention_heads: int = 6,
        device: Optional[str] = None
    ):
        """
        Initialize the distiller.
        
        Args:
            teacher_model_path: Path to the teacher E5 model
            output_dir: Directory to save the distilled model
            hidden_size: Hidden size for student model
            hidden_layers: Number of layers for student model
            attention_heads: Number of attention heads for student model
            device: Device to use (cuda/cpu)
        """
        self.teacher_model_path = teacher_model_path
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(teacher_model_path)
        
        # Initialize teacher model
        print("Loading teacher model...")
        self.teacher_model = SentenceTransformer(teacher_model_path, device=self.device)
        self.teacher_model.eval()  # Set to evaluation mode
        
        # Initialize student model
        print("Creating student model...")
        self.student_model = E5StudentModel(
            teacher_model_path=teacher_model_path,
            hidden_size=hidden_size,
            hidden_layers=hidden_layers,
            attention_heads=attention_heads
        ).to(self.device)
        
        # Initialize model size info
        self._calculate_model_sizes()
    
    def _calculate_model_sizes(self):
        """Calculate and store model sizes for comparison"""
        # Calculate teacher model size
        teacher_params = sum(p.numel() for p in self.teacher_model.parameters())
        
        # Calculate student model size
        student_params = sum(p.numel() for p in self.student_model.parameters())
        
        # Store sizes
        self.teacher_size = teacher_params
        self.student_size = student_params
        
        print(f"Teacher model parameters: {teacher_params:,}")
        print(f"Student model parameters: {student_params:,}")
        print(f"Size reduction: {(1 - student_params/teacher_params) * 100:.2f}%")
    
    def prepare_dataset(
        self, 
        texts: List[str], 
        val_split: float = 0.1,
        batch_size: int = 32,
        num_workers: int = 4
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare dataset for distillation by computing teacher embeddings.
        
        Args:
            texts: List of texts for training
            val_split: Validation split ratio
            batch_size: Batch size for data loaders
            num_workers: Number of workers for data loaders
            
        Returns:
            Tuple of (train_dataloader, val_dataloader)
        """
        print("Computing teacher embeddings...")
        
        # Compute teacher embeddings for all texts
        with torch.no_grad():
            # Format texts with prefixes
            formatted_texts = []
            for text in texts:
                if not text.startswith(("query: ", "passage: ")):
                    if len(text.strip().split()) <= 10:
                        formatted_texts.append(f"query: {text}")
                    else:
                        formatted_texts.append(f"passage: {text}")
                else:
                    formatted_texts.append(text)
            
            # Compute embeddings in batches
            teacher_embeddings = self.teacher_model.encode(
                formatted_texts,
                batch_size=batch_size,
                show_progress_bar=True,
                normalize_embeddings=True
            )
        
        # Split into train and validation
        n_samples = len(texts)
        n_val = int(n_samples * val_split)
        n_train = n_samples - n_val
        
        # Shuffle indices
        indices = np.random.permutation(n_samples)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        # Create datasets
        train_dataset = E5DistillationDataset(
            [texts[i] for i in train_indices],
            self.tokenizer,
            teacher_embeddings[train_indices]
        )
        
        val_dataset = E5DistillationDataset(
            [texts[i] for i in val_indices],
            self.tokenizer,
            teacher_embeddings[val_indices]
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        
        return train_loader, val_loader
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 5,
        learning_rate: float = 3e-5,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
        save_every: int = 1
    ) -> Dict:
        """
        Train the student model via knowledge distillation.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            epochs: Number of training epochs
            learning_rate: Learning rate
            warmup_steps: Steps for LR warmup
            weight_decay: Weight decay for optimizer
            save_every: Save model every N epochs
            
        Returns:
            Dictionary with training history
        """
        # Set student model to training mode
        self.student_model.train()
        
        # Loss function - MSE for cosine similarity in normalized embedding space
        loss_fn = nn.MSELoss()
        
        # Optimizer
        optimizer = AdamW(
            self.student_model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=len(train_loader) * epochs
        )
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_similarity': []
        }
        
        # Training loop
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Training
            train_losses = []
            train_pbar = tqdm(train_loader, desc=f"Training")
            
            for batch in train_pbar:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                teacher_embedding = batch['teacher_embedding'].to(self.device)
                
                # Forward pass
                student_embedding = self.student_model(input_ids, attention_mask)
                
                # Compute loss
                loss = loss_fn(student_embedding, teacher_embedding)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Update parameters
                optimizer.step()
                scheduler.step()
                
                # Update progress bar
                train_losses.append(loss.item())
                train_pbar.set_postfix({'loss': np.mean(train_losses[-100:])})
            
            # Calculate average training loss
            avg_train_loss = np.mean(train_losses)
            history['train_loss'].append(avg_train_loss)
            
            # Validation
            self.student_model.eval()
            val_losses = []
            similarities = []
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f"Validation")
                
                for batch in val_pbar:
                    # Move batch to device
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    teacher_embedding = batch['teacher_embedding'].to(self.device)
                    
                    # Forward pass
                    student_embedding = self.student_model(input_ids, attention_mask)
                    
                    # Compute loss
                    loss = loss_fn(student_embedding, teacher_embedding)
                    val_losses.append(loss.item())
                    
                    # Compute cosine similarity
                    for i in range(len(student_embedding)):
                        sim = F.cosine_similarity(
                            student_embedding[i].unsqueeze(0),
                            teacher_embedding[i].unsqueeze(0)
                        ).item()
                        similarities.append(sim)
                    
                    # Update progress bar
                    val_pbar.set_postfix({'loss': np.mean(val_losses[-100:])})
            
            # Calculate average validation metrics
            avg_val_loss = np.mean(val_losses)
            avg_similarity = np.mean(similarities)
            
            history['val_loss'].append(avg_val_loss)
            history['val_similarity'].append(avg_similarity)
            
            print(f"Train Loss: {avg_train_loss:.6f}")
            print(f"Val Loss: {avg_val_loss:.6f}")
            print(f"Val Similarity: {avg_similarity:.6f}")
            
            # Save model checkpoint
            if (epoch + 1) % save_every == 0:
                checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-{epoch+1}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                
                # Save model
                torch.save(self.student_model.state_dict(), os.path.join(checkpoint_dir, "model.pt"))
                
                # Save tokenizer
                self.tokenizer.save_pretrained(checkpoint_dir)
                
                print(f"Saved checkpoint to {checkpoint_dir}")
            
            # Set back to training mode
            self.student_model.train()
        
        # Save final model
        self.save_model()
        
        return history
    
    def save_model(self, model_name: str = "final"):
        """
        Save the trained student model.
        
        Args:
            model_name: Name for the saved model
        """
        model_dir = os.path.join(self.output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model state
        torch.save(self.student_model.state_dict(), os.path.join(model_dir, "model.pt"))
        
        # Save tokenizer
        self.tokenizer.save_pretrained(model_dir)
        
        # Save config
        student_config = {
            "teacher_model_path": self.teacher_model_path,
            "model_type": "E5Student",
            "hidden_size": self.student_model.model.config.hidden_size,
            "num_hidden_layers": self.student_model.model.config.num_hidden_layers,
            "num_attention_heads": self.student_model.model.config.num_attention_heads,
            "intermediate_size": self.student_model.model.config.intermediate_size,
            "teacher_embedding_dim": self.student_model.teacher_dim,
            "has_projection": self.student_model.projection is not None,
            "size_reduction_percentage": (1 - self.student_size/self.teacher_size) * 100
        }
        
        with open(os.path.join(model_dir, "config.json"), "w") as f:
            json.dump(student_config, f, indent=2)
        
        print(f"Model saved to {model_dir}")
    
    def plot_training_history(self, history: Dict, output_file: str = None):
        """
        Plot training history.
        
        Args:
            history: Training history dictionary
            output_file: Path to save the plot
        """
        plt.figure(figsize=(12, 8))
        
        # Plot losses
        plt.subplot(2, 1, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Loss During Training')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot similarity
        plt.subplot(2, 1, 2)
        plt.plot(history['val_similarity'], label='Validation Similarity', color='green')
        plt.title('Embedding Similarity to Teacher')
        plt.xlabel('Epoch')
        plt.ylabel('Cosine Similarity')
        plt.axhline(y=0.9, linestyle='--', color='red', alpha=0.7, label='0.9 Threshold')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save or show plot
        if output_file:
            plt.savefig(output_file)
            print(f"Training history plot saved to {output_file}")
        else:
            plt.show()
    
    def measure_latency(
        self,
        texts: List[str],
        batch_size: int = 32,
        n_runs: int = 5
    ) -> Dict:
        """
        Measure and compare latency between teacher and student models.
        
        Args:
            texts: List of text samples to encode
            batch_size: Batch size for encoding
            n_runs: Number of runs for statistical significance
            
        Returns:
            Dictionary with latency measurements
        """
        results = {
            "teacher": {"single": [], "batch": []},
            "student": {"single": [], "batch": []}
        }
        
        # Format texts with prefixes
        formatted_texts = []
        for text in texts:
            if not text.startswith(("query: ", "passage: ")):
                if len(text.strip().split()) <= 10:
                    formatted_texts.append(f"query: {text}")
                else:
                    formatted_texts.append(f"passage: {text}")
            else:
                formatted_texts.append(text)
        
        # Set both models to eval mode
        self.student_model.eval()
        self.teacher_model.eval()
        
        print("Measuring teacher model latency...")
        # Measure teacher model latency
        for _ in range(n_runs):
            # Single query latency
            for text in tqdm(formatted_texts[:10], desc="Single query"):  # Use subset for single
                start_time = time.time()
                _ = self.teacher_model.encode(text, normalize_embeddings=True)
                end_time = time.time()
                latency = (end_time - start_time) * 1000  # Convert to ms
                results["teacher"]["single"].append(latency)
            
            # Batch latency
            start_time = time.time()
            _ = self.teacher_model.encode(
                formatted_texts, 
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            end_time = time.time()
            total_time = (end_time - start_time) * 1000  # ms
            per_query_time = total_time / len(formatted_texts)
            results["teacher"]["batch"].append(per_query_time)
        
        print("Measuring student model latency...")
        # Measure student model latency
        with torch.no_grad():
            for _ in range(n_runs):
                # Single query latency
                for text in tqdm(formatted_texts[:10], desc="Single query"):  # Use subset for single
                    # Tokenize
                    inputs = self.tokenizer(
                        text,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    start_time = time.time()
                    _ = self.student_model(inputs["input_ids"], inputs["attention_mask"])
                    end_time = time.time()
                    
                    latency = (end_time - start_time) * 1000  # Convert to ms
                    results["student"]["single"].append(latency)
                
                # Batch latency
                all_latencies = []
                
                # Process in batches
                for i in range(0, len(formatted_texts), batch_size):
                    batch_texts = formatted_texts[i:i+batch_size]
                    
                    # Tokenize
                    inputs = self.tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    start_time = time.time()
                    _ = self.student_model(inputs["input_ids"], inputs["attention_mask"])
                    end_time = time.time()
                    
                    total_time = (end_time - start_time) * 1000  # ms
                    per_query_time = total_time / len(batch_texts)
                    all_latencies.append(per_query_time)
                
                # Average across all batches
                avg_latency = sum(all_latencies) / len(all_latencies)
                results["student"]["batch"].append(avg_latency)
        
        # Calculate statistics
        stats = {}
        
        for model in ["teacher", "student"]:
            stats[model] = {}
            
            for mode in ["single", "batch"]:
                data = results[model][mode]
                stats[model][mode] = {
                    "mean": np.mean(data),
                    "std": np.std(data),
                    "min": np.min(data),
                    "max": np.max(data),
                    "p50": np.percentile(data, 50),
                    "p95": np.percentile(data, 95),
                    "p99": np.percentile(data, 99)
                }
        
        # Calculate speedup
        teacher_single_mean = stats["teacher"]["single"]["mean"]
        student_single_mean = stats["student"]["single"]["mean"]
        single_speedup = teacher_single_mean / student_single_mean
        
        teacher_batch_mean = stats["teacher"]["batch"]["mean"]
        student_batch_mean = stats["student"]["batch"]["mean"]
        batch_speedup = teacher_batch_mean / student_batch_mean
        
        stats["speedup"] = {
            "single": single_speedup,
            "batch": batch_speedup
        }
        
        return stats

def load_medical_queries(query_file: str = None) -> List[str]:
    """
    Load medical queries from a file or use defaults.
    
    Args:
        query_file: Path to a JSON or text file with medical queries
        
    Returns:
        List of medical query strings
    """
    default_queries = [
        "What are the symptoms of Type 2 Diabetes?",
        "How is hypertension diagnosed?",
        "What are common side effects of statins?",
        "Can you describe the pathophysiology of heart failure?",
        "What's the difference between CT and MRI imaging?",
        "How does chemotherapy work for cancer treatment?",
        "What are the latest treatments for rheumatoid arthritis?",
        "Explain the mechanism of action for ACE inhibitors",
        "What causes chronic kidney disease?",
        "How is multiple sclerosis diagnosed and treated?",
        "What are the risk factors for developing Alzheimer's disease?",
        "How does insulin resistance lead to Type 2 Diabetes?",
        "What are the stages of chronic obstructive pulmonary disease?",
        "Describe the pathophysiology of asthma attacks",
        "What are common complications of untreated hypothyroidism?"
    ]
    
    if query_file is None:
        return default_queries
    
    # Load queries from file if provided
    try:
        ext = os.path.splitext(query_file)[1].lower()
        if ext == '.json':
            with open(query_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict) and 'queries' in data:
                    return data['queries']
                else:
                    print(f"Warning: Couldn't parse JSON format. Using default queries.")
                    return default_queries
        else:
            # Assume text file with one query per line
            with open(query_file, 'r') as f:
                queries = [line.strip() for line in f if line.strip()]
                return queries
    except Exception as e:
        print(f"Error loading queries file: {e}")
        print("Using default medical queries instead.")
        return default_queries

def generate_synthetic_data(seed_queries: List[str], n_samples: int = 1000) -> List[str]:
    """
    Generate synthetic data for training by augmenting seed queries.
    
    Args:
        seed_queries: List of seed medical queries
        n_samples: Number of samples to generate
        
    Returns:
        List of synthetic texts
    """
    # Simple augmentation techniques
    augmented_data = []
    augmented_data.extend(seed_queries)  # Include original samples
    
    # Medical prefixes to add variety
    prefixes = [
        "Can you explain ", "What is ", "How does ", "Describe ", 
        "What are symptoms of ", "What causes ", "How to treat ",
        "What are risk factors for ", "How to diagnose ", "What is the pathophysiology of ",
    ]
    
    # Medical conditions and topics
    medical_topics = [
        "diabetes", "hypertension", "heart disease", "cancer", "asthma",
        "arthritis", "Alzheimer's disease", "Parkinson's disease", "multiple sclerosis",
        "kidney disease", "liver disease", "thyroid disorders", "depression",
        "anxiety", "stroke", "COPD", "pneumonia", "bronchitis",
        "gastritis", "IBS", "Crohn's disease", "ulcerative colitis"
    ]
    
    # Generate combinations
    from itertools import product
    import random
    random.seed(42)  # For reproducibility
    
    combinations = list(product(prefixes, medical_topics))
    random.shuffle(combinations)
    
    # Generate queries
    for prefix, topic in combinations[:n_samples - len(augmented_data)]:
        query = f"{prefix}{topic}?"
        if query not in augmented_data:
            augmented_data.append(query)
    
    # If we still need more data, add variations
    if len(augmented_data) < n_samples:
        medical_adjectives = ["chronic", "acute", "severe", "mild", "recurring", "treatment-resistant"]
        
        for adj in medical_adjectives:
            for topic in medical_topics:
                if len(augmented_data) >= n_samples:
                    break
                    
                query = f"What is {adj} {topic}?"
                if query not in augmented_data:
                    augmented_data.append(query)
    
    return augmented_data[:n_samples]

def main():
    parser = argparse.ArgumentParser(description="Knowledge Distillation for E5 Model")
    parser.add_argument("--teacher_model_path", type=str, required=True, 
                        help="Path to the teacher E5 model")
    parser.add_argument("--output_dir", type=str, default="./distilled_e5",
                        help="Output directory for distilled model")
    parser.add_argument("--data_file", type=str, default=None,
                        help="Path to training data file (JSON or TXT)")
    parser.add_argument("--synthetic_data", action="store_true",
                        help="Generate synthetic data for training")
    parser.add_argument("--n_samples", type=int, default=1000,
                        help="Number of synthetic samples to generate")
    parser.add_argument("--hidden_size", type=int, default=384,
                        help="Hidden size for student model")
    parser.add_argument("--hidden_layers", type=int, default=6,
                        help="Number of hidden layers for student model")
    parser.add_argument("--attention_heads", type=int, default=6,
                        help="Number of attention heads for student model")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=3e-5,
                        help="Learning rate")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--eval_only", action="store_true",
                        help="Only evaluate latency without training")
    parser.add_argument("--load_model", type=str, default=None,
                        help="Path to load a previously distilled model")
    
    args = parser.parse_args()
    
    # Initialize distiller
    distiller = E5Distiller(
        teacher_model_path=args.teacher_model_path,
        output_dir=args.output_dir,
        hidden_size=args.hidden_size,
        hidden_layers=args.hidden_layers,
        attention_heads=args.attention_heads,
        device=args.device
    )
    
    # Load previously distilled model if specified
    if args.load_model:
        print(f"Loading distilled model from {args.load_model}")
        distiller.student_model.load_state_dict(torch.load(
            os.path.join(args.load_model, "model.pt"),
            map_location=distiller.device
        ))
    
    # Load or generate training data
    if args.eval_only:
        # For evaluation only, use default queries
        texts = load_medical_queries(args.data_file)
        print(f"Loaded {len(texts)} queries for evaluation")
        
        # Measure latency
        latency_stats = distiller.measure_latency(
            texts=texts,
            batch_size=args.batch_size
        )
        
        # Print latency comparison
        print("\n" + "="*60)
        print("LATENCY COMPARISON: TEACHER vs STUDENT")
        print("="*60)
        
        print("\nSingle Query Latency (ms):")
        print(f"  Teacher: {latency_stats['teacher']['single']['mean']:.2f} ms")
        print(f"  Student: {latency_stats['student']['single']['mean']:.2f} ms")
        print(f"  Speedup: {latency_stats['speedup']['single']:.2f}x")
        
        print("\nBatch Processing Latency (ms per query):")
        print(f"  Teacher: {latency_stats['teacher']['batch']['mean']:.2f} ms")
        print(f"  Student: {latency_stats['student']['batch']['mean']:.2f} ms")
        print(f"  Speedup: {latency_stats['speedup']['batch']:.2f}x")
        
        print("\nP95 Latency (ms):")
        print(f"  Teacher: {latency_stats['teacher']['single']['p95']:.2f} ms")
        print(f"  Student: {latency_stats['student']['single']['p95']:.2f} ms")
        
        print("\nModel Size Reduction:")
        print(f"  Teacher parameters: {distiller.teacher_size:,}")
        print(f"  Student parameters: {distiller.student_size:,}")
        print(f"  Size reduction: {(1 - distiller.student_size/distiller.teacher_size) * 100:.2f}%")
        
        # Save latency comparison to file
        output_file = os.path.join(args.output_dir, "latency_comparison.json")
        with open(output_file, "w") as f:
            json.dump(latency_stats, f, indent=2)
        
        print(f"\nLatency comparison saved to {output_file}")
    else:
        # For training, get data from file or generate synthetic data
        if args.data_file:
            texts = load_medical_queries(args.data_file)
            print(f"Loaded {len(texts)} texts from {args.data_file}")
        elif args.synthetic_data:
            seed_queries = load_medical_queries()
            texts = generate_synthetic_data(seed_queries, args.n_samples)
            print(f"Generated {len(texts)} synthetic texts for training")
        else:
            print("Error: Must provide --data_file or --synthetic_data")
            return
        
        # Prepare dataset
        train_loader, val_loader = distiller.prepare_dataset(
            texts=texts,
            batch_size=args.batch_size,
            val_split=0.1
        )
        
        # Train student model
        history = distiller.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            save_every=1
        )
        
        # Plot training history
        history_plot = os.path.join(args.output_dir, "training_history.png")
        distiller.plot_training_history(history, history_plot)
        
        # Measure latency
        latency_stats = distiller.measure_latency(
            texts=texts[:100],  # Use subset for speed
            batch_size=args.batch_size
        )
        
        # Print latency comparison
        print("\n" + "="*60)
        print("TRAINING COMPLETE - LATENCY COMPARISON")
        print("="*60)
        
        print(f"\nSingle Query Speedup: {latency_stats['speedup']['single']:.2f}x")
        print(f"Batch Processing Speedup: {latency_stats['speedup']['batch']:.2f}x")
        print(f"Model Size Reduction: {(1 - distiller.student_size/distiller.teacher_size) * 100:.2f}%")
        
        # Save latency comparison to file
        output_file = os.path.join(args.output_dir, "latency_comparison.json")
        with open(output_file, "w") as f:
            json.dump(latency_stats, f, indent=2)
        
        print(f"\nDistilled model saved to {args.output_dir}")
        print(f"Latency comparison saved to {output_file}")

if __name__ == "__main__":
    main()