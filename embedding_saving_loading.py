import pickle
import numpy as np
import json
from pathlib import Path
import gzip
import joblib
from typing import Dict, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ICDEmbeddingsIO:
    """Efficient saving and loading of ICD code embeddings dictionary."""
    
    @staticmethod
    def save_pickle(embeddings_dict: Dict[str, np.ndarray], filepath: str, compress: bool = True) -> None:
        """
        Save embeddings dictionary using pickle with optional compression.
        
        Args:
            embeddings_dict: Dictionary with ICD descriptions as keys and embeddings as values
            filepath: Path to save the file
            compress: Whether to use gzip compression (recommended for large files)
        """
        filepath = Path(filepath)
        
        try:
            if compress:
                with gzip.open(f"{filepath}.pkl.gz", 'wb') as f:
                    pickle.dump(embeddings_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(f"Compressed embeddings saved to {filepath}.pkl.gz")
            else:
                with open(f"{filepath}.pkl", 'wb') as f:
                    pickle.dump(embeddings_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(f"Embeddings saved to {filepath}.pkl")
                
        except Exception as e:
            logger.error(f"Error saving embeddings: {e}")
            raise
    
    @staticmethod
    def load_pickle(filepath: str) -> Dict[str, np.ndarray]:
        """
        Load embeddings dictionary from pickle file.
        
        Args:
            filepath: Path to the saved file (with or without extension)
            
        Returns:
            Dictionary with ICD descriptions as keys and embeddings as values
        """
        filepath = Path(filepath)
        
        # Try compressed file first
        compressed_path = filepath.with_suffix('.pkl.gz') if filepath.suffix != '.gz' else filepath
        regular_path = filepath.with_suffix('.pkl') if filepath.suffix != '.pkl' else filepath
        
        try:
            if compressed_path.exists():
                with gzip.open(compressed_path, 'rb') as f:
                    embeddings_dict = pickle.load(f)
                logger.info(f"Loaded compressed embeddings from {compressed_path}")
                return embeddings_dict
            elif regular_path.exists():
                with open(regular_path, 'rb') as f:
                    embeddings_dict = pickle.load(f)
                logger.info(f"Loaded embeddings from {regular_path}")
                return embeddings_dict
            else:
                raise FileNotFoundError(f"No embeddings file found at {filepath}")
                
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            raise
    
    @staticmethod
    def save_joblib(embeddings_dict: Dict[str, np.ndarray], filepath: str, compress: int = 3) -> None:
        """
        Save embeddings dictionary using joblib (good for numpy arrays).
        
        Args:
            embeddings_dict: Dictionary with ICD descriptions as keys and embeddings as values
            filepath: Path to save the file
            compress: Compression level (0-9, 3 is good balance of speed/size)
        """
        filepath = Path(filepath).with_suffix('.joblib')
        
        try:
            joblib.dump(embeddings_dict, filepath, compress=compress)
            logger.info(f"Embeddings saved with joblib to {filepath}")
        except Exception as e:
            logger.error(f"Error saving embeddings with joblib: {e}")
            raise
    
    @staticmethod
    def load_joblib(filepath: str) -> Dict[str, np.ndarray]:
        """
        Load embeddings dictionary from joblib file.
        
        Args:
            filepath: Path to the saved file
            
        Returns:
            Dictionary with ICD descriptions as keys and embeddings as values
        """
        filepath = Path(filepath).with_suffix('.joblib')
        
        try:
            embeddings_dict = joblib.load(filepath)
            logger.info(f"Loaded embeddings with joblib from {filepath}")
            return embeddings_dict
        except Exception as e:
            logger.error(f"Error loading embeddings with joblib: {e}")
            raise
    
    @staticmethod
    def save_npz(embeddings_dict: Dict[str, np.ndarray], filepath: str) -> None:
        """
        Save embeddings using numpy's compressed format.
        Efficient for large numpy arrays but keys must be valid Python identifiers.
        
        Args:
            embeddings_dict: Dictionary with ICD descriptions as keys and embeddings as values
            filepath: Path to save the file
        """
        filepath = Path(filepath).with_suffix('.npz')
        
        try:
            # Convert keys to valid numpy savez keys and save mapping
            key_mapping = {f"embed_{i}": key for i, key in enumerate(embeddings_dict.keys())}
            reverse_mapping = {v: k for k, v in key_mapping.items()}
            
            # Prepare data for numpy
            np_dict = {reverse_mapping[key]: value for key, value in embeddings_dict.items()}
            
            # Save both embeddings and key mapping
            np.savez_compressed(filepath, key_mapping=np.array(list(key_mapping.items()), dtype=object), **np_dict)
            logger.info(f"Embeddings saved with numpy to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving embeddings with numpy: {e}")
            raise
    
    @staticmethod
    def load_npz(filepath: str) -> Dict[str, np.ndarray]:
        """
        Load embeddings dictionary from numpy compressed file.
        
        Args:
            filepath: Path to the saved file
            
        Returns:
            Dictionary with ICD descriptions as keys and embeddings as values
        """
        filepath = Path(filepath).with_suffix('.npz')
        
        try:
            data = np.load(filepath, allow_pickle=True)
            key_mapping = dict(data['key_mapping'])
            
            embeddings_dict = {}
            for np_key, original_key in key_mapping.items():
                embeddings_dict[original_key] = data[np_key]
            
            logger.info(f"Loaded embeddings with numpy from {filepath}")
            return embeddings_dict
            
        except Exception as e:
            logger.error(f"Error loading embeddings with numpy: {e}")
            raise

# Convenience functions for quick usage
def save_icd_embeddings(embeddings_dict: Dict[str, np.ndarray], filepath: str, method: str = 'pickle') -> None:
    """
    Quick save function with method selection.
    
    Args:
        embeddings_dict: Dictionary to save
        filepath: Path to save file
        method: 'pickle', 'joblib', or 'npz'
    """
    io_handler = ICDEmbeddingsIO()
    
    if method == 'pickle':
        io_handler.save_pickle(embeddings_dict, filepath, compress=True)
    elif method == 'joblib':
        io_handler.save_joblib(embeddings_dict, filepath)
    elif method == 'npz':
        io_handler.save_npz(embeddings_dict, filepath)
    else:
        raise ValueError("Method must be 'pickle', 'joblib', or 'npz'")

def load_icd_embeddings(filepath: str, method: str = 'pickle') -> Dict[str, np.ndarray]:
    """
    Quick load function with method selection.
    
    Args:
        filepath: Path to saved file
        method: 'pickle', 'joblib', or 'npz'
        
    Returns:
        Dictionary with ICD descriptions as keys and embeddings as values
    """
    io_handler = ICDEmbeddingsIO()
    
    if method == 'pickle':
        return io_handler.load_pickle(filepath)
    elif method == 'joblib':
        return io_handler.load_joblib(filepath)
    elif method == 'npz':
        return io_handler.load_npz(filepath)
    else:
        raise ValueError("Method must be 'pickle', 'joblib', or 'npz'")

# Example usage
if __name__ == "__main__":
    # Example embeddings dictionary
    sample_embeddings = {
        "A00-A09 Intestinal infectious diseases": np.random.rand(768),
        "A15-A19 Tuberculosis": np.random.rand(768),
        "A20-A28 Certain zoonotic bacterial diseases": np.random.rand(768),
    }
    
    # Method 1: Pickle (recommended for general use)
    print("Testing pickle method...")
    save_icd_embeddings(sample_embeddings, "icd_embeddings", method="pickle")
    loaded_embeddings = load_icd_embeddings("icd_embeddings", method="pickle")
    print(f"Loaded {len(loaded_embeddings)} embeddings with pickle")
    
    # Method 2: Joblib (good for numpy arrays)
    print("\nTesting joblib method...")
    save_icd_embeddings(sample_embeddings, "icd_embeddings", method="joblib")
    loaded_embeddings = load_icd_embeddings("icd_embeddings", method="joblib")
    print(f"Loaded {len(loaded_embeddings)} embeddings with joblib")
    
    # Method 3: Numpy compressed (most efficient for large arrays)
    print("\nTesting npz method...")
    save_icd_embeddings(sample_embeddings, "icd_embeddings", method="npz")
    loaded_embeddings = load_icd_embeddings("icd_embeddings", method="npz")
    print(f"Loaded {len(loaded_embeddings)} embeddings with npz")
    
    print("\nAll methods tested successfully!")
