#Usage Examples

# Basic usage with standard transformers
# python convert_to_fp16.py --model_path "/path/to/your/finetuned-e5-large"

# Using SentenceTransformers implementation and custom output path
# python convert_to_fp16.py --model_path "/path/to/your/finetuned-e5-large" --output_path "/path/to/save/fp16-model" --use_sentence_transformer

# Convert and verify the model
# python convert_to_fp16.py --model_path "/path/to/your/finetuned-e5-large" --verify

import argparse
import os
import torch
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer

def save_model_fp16(model_path, output_path, use_sentence_transformer=False):
    """
    Load an E5 model and save it in FP16 format
    
    Args:
        model_path (str): Path to the original E5 model
        output_path (str): Path to save the FP16 model
        use_sentence_transformer (bool): Whether to use SentenceTransformer or standard transformers
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Load the model based on implementation preference
    print(f"Loading model from {model_path}")
    if use_sentence_transformer:
        model = SentenceTransformer(model_path)
        
        # Access the underlying transformer model
        for module in model.modules():
            if hasattr(module, 'auto_model'):
                transformer_model = module.auto_model
                transformer_tokenizer = module.tokenizer
                break
    else:
        transformer_model = AutoModel.from_pretrained(model_path)
        transformer_tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Convert model to half precision (FP16)
    print("Converting model to FP16 format")
    transformer_model = transformer_model.half()  # Convert to FP16
    
    # Save the tokenizer
    print(f"Saving tokenizer to {output_path}")
    transformer_tokenizer.save_pretrained(output_path)
    
    # Save the model in FP16 format
    print(f"Saving model to {output_path}")
    transformer_model.save_pretrained(output_path)
    
    # If using sentence_transformer, save the full ST model with config
    if use_sentence_transformer:
        print(f"Saving SentenceTransformer config to {output_path}")
        model.save(output_path)
    
    # Verify the model sizes
    original_size = get_model_size(model_path)
    converted_size = get_model_size(output_path)
    
    print(f"\nOriginal model size: {original_size:.2f} MB")
    print(f"FP16 model size: {converted_size:.2f} MB")
    print(f"Size reduction: {(1 - converted_size / original_size) * 100:.2f}%")
    
    return output_path

def get_model_size(model_path):
    """Get the size of the model in MB"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(model_path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total_size += os.path.getsize(filepath)
    return total_size / (1024 * 1024)  # Convert to MB

def verify_model(model_path, test_text="This is a test sentence for verification."):
    """Verify that the model can be loaded and used for inference"""
    try:
        # Try to load as SentenceTransformer first
        try:
            model = SentenceTransformer(model_path)
            print("Successfully loaded as SentenceTransformer")
            embeddings = model.encode([f"query: {test_text}"])
            print(f"Generated embeddings shape: {embeddings.shape}")
            return True
        except Exception as e:
            print(f"Failed to load as SentenceTransformer: {e}")
            
            # Try to load as standard transformer
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModel.from_pretrained(model_path)
            print("Successfully loaded as standard transformer")
            
            # Generate embeddings
            inputs = tokenizer(f"query: {test_text}", return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0]
            print(f"Generated embeddings shape: {embeddings.shape}")
            return True
            
    except Exception as e:
        print(f"Failed to verify model: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Convert E5 model to FP16 format")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the original E5 model")
    parser.add_argument("--output_path", type=str, help="Path to save the FP16 model")
    parser.add_argument("--use_sentence_transformer", action="store_true", 
                      help="Use SentenceTransformer implementation")
    parser.add_argument("--verify", action="store_true", 
                      help="Verify the converted model by running inference")
    args = parser.parse_args()
    
    # Set default output path if not provided
    if args.output_path is None:
        args.output_path = args.model_path + "-fp16"
    
    # Convert and save the model in FP16
    output_path = save_model_fp16(
        args.model_path, 
        args.output_path, 
        args.use_sentence_transformer
    )
    
    # Verify the model if requested
    if args.verify:
        print("\nVerifying converted model...")
        success = verify_model(output_path)
        if success:
            print("Model verification successful!")
        else:
            print("Model verification failed!")

if __name__ == "__main__":
    main()