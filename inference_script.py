import argparse
import torch
from typing import List, Union, Dict
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer

class E5Embedder:
    """Base class for E5 embedding models"""
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
    def embed(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """Generate embeddings for the input text(s)"""
        raise NotImplementedError("Subclasses must implement this method")

class TransformerE5Embedder(E5Embedder):
    """Implementation using HuggingFace Transformers directly"""
    def __init__(self, model_path: str):
        super().__init__(model_path)
        print(f"Loading tokenizer from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"Loading model from {model_path}")
        self.model = AutoModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
    def embed(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """Generate embeddings using the transformer model directly"""
        # Handle single text input
        if isinstance(texts, str):
            texts = [texts]
            
        # E5 models expect inputs with format: "query: " + text
        formatted_texts = [f"query: {text}" for text in texts]
        
        # Tokenize the texts
        encoded_input = self.tokenizer(
            formatted_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            
        # Use the [CLS] token embedding (first token)
        embeddings = model_output.last_hidden_state[:, 0]
        
        # Normalize embeddings to unit length
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings

class SentenceTransformerE5Embedder(E5Embedder):
    """Implementation using SentenceTransformers"""
    def __init__(self, model_path: str):
        super().__init__(model_path)
        print(f"Loading SentenceTransformer from {model_path}")
        self.model = SentenceTransformer(model_path)
        self.model.to(self.device)
        
    def embed(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """Generate embeddings using SentenceTransformer"""
        # Handle single text input
        if isinstance(texts, str):
            texts = [texts]
            
        # E5 models expect inputs with format: "query: " + text
        formatted_texts = [f"query: {text}" for text in texts]
        
        # Generate embeddings
        embeddings = self.model.encode(
            formatted_texts, 
            convert_to_tensor=True,
            show_progress_bar=True if len(texts) > 10 else False
        )
        
        return embeddings


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings using a fine-tuned E5 model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned E5 model")
    parser.add_argument("--implementation", type=str, choices=["transformers", "sentence_transformers"], 
                        default="sentence_transformers", help="Implementation to use")
    parser.add_argument("--input", type=str, help="Text to generate embeddings for")
    parser.add_argument("--input_file", type=str, help="Path to a file with one text per line")
    args = parser.parse_args()
    
    # Validate input arguments
    if args.input is None and args.input_file is None:
        parser.error("Either --input or --input_file must be provided")
    
    # Create the appropriate embedder
    if args.implementation == "transformers":
        embedder = TransformerE5Embedder(args.model_path)
    else:  # sentence_transformers
        embedder = SentenceTransformerE5Embedder(args.model_path)
    
    # Get the text input
    if args.input:
        texts = [args.input]
    else:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
    
    # Generate embeddings
    print(f"Generating embeddings for {len(texts)} text(s)...")
    embeddings = embedder.embed(texts)
    
    # Print results
    for i, (text, embedding) in enumerate(zip(texts, embeddings)):
        print(f"\nText {i+1}: {text[:50]}{'...' if len(text) > 50 else ''}")
        print(f"Embedding shape: {embedding.shape}")
        print(f"Embedding (first 5 dimensions): {embedding[:5].cpu().numpy()}")
        print(f"Embedding norm: {torch.norm(embedding).item()}")
    
    print(f"\nTotal embeddings generated: {len(embeddings)}")
    print(f"Embedding dimension: {embeddings[0].shape[0]}")

if __name__ == "__main__":
    main()