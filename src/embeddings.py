import json
import os
import torch
import pandas as pd
from transformers import AutoImageProcessor, AutoModel
from datasets import load_dataset
from PIL import Image
import numpy as np
from typing import Dict, Any, List
from tqdm import tqdm


class EmbeddingGenerator:
    """Generate image embeddings using foundation models."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.embeddings_config = config['embeddings']
        self.model_name = self.embeddings_config['model_name']
        self.output_file = self.embeddings_config['output_file']
        
        # Load model and processor
        print(f"Loading model: {self.model_name}")
        self.processor = AutoImageProcessor.from_pretrained(self.model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(self.model_name)
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        print(f"Using device: {self.device}")

    def generate_embeddings(self, image: Image.Image) -> List[float]:
        """Generate embeddings for a single image."""
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Extract embeddings (using CLS token for ViT models)
        if hasattr(outputs, 'last_hidden_state'):
            embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
        else:
            embeddings = outputs.pooler_output
            
        return embeddings.cpu().squeeze().tolist()

    def append_to_json(self, data: Dict[str, Any]):
        """Append data to JSON file."""
        with open(self.output_file, 'a') as f:
            json.dump(data, f)
            f.write('\n')

    def process_dataset(self, dataset_name: str, start_index: int = 0):
        """Process entire dataset and generate embeddings."""
        print(f"Loading dataset: {dataset_name}")
        dataset = load_dataset(dataset_name, split='train')
        df = pd.DataFrame(dataset)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        
        # Initialize or clear output file
        if start_index == 0:
            open(self.output_file, 'w').close()
        
        print(f"Processing {len(df)} images starting from index {start_index}")
        
        for idx in tqdm(range(start_index, len(df)), desc="Generating embeddings"):
            try:
                image = df.iloc[idx]['Raw_images']
                if image is None:
                    print(f"No image found for index {idx}")
                    continue
                    
                embedding = self.generate_embeddings(image)
                
                entry = {
                    "index": idx,
                    "embedding": embedding
                }
                
                self.append_to_json(entry)
                
                if idx % 100 == 0:
                    print(f"Processed {idx} images")
                    
            except Exception as e:
                print(f"Error processing index {idx}: {e}")
                continue
        
        print(f"Embeddings saved to {self.output_file}")

    def load_embeddings(self) -> List[List[float]]:
        """Load embeddings from file."""
        embeddings = []
        with open(self.output_file, 'r') as f:
            for line in f:
                entry = json.loads(line)
                embeddings.append(entry['embedding'])
        return embeddings