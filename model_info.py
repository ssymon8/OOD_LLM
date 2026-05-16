import os
import sys
import torch
from torchinfo import summary
import json
from pathlib import Path
from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend, FineGrainedFP8Config

import logging

"""
Helper script to view the Mistral 3 model architecture.
"""

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Script started!")
    try:
        model_path = os.path.expanduser("~/OOD_LLM/ministral")
        
        # Verify model path exists
        if not Path(model_path).exists():
            logger.error(f"Model path not found: {model_path}")
            sys.exit(1)
        
        # Detect device
        if not torch.cuda.is_available():
            logger.error("CUDA not available :/")
            sys.exit(1)
        device = "cuda"
        logger.info(f"Using device: {device}")
        
        logger.info(f"Loading model from {model_path}...")
        
        # Model loading
        model = Mistral3ForConditionalGeneration.from_pretrained(
            model_path,
            device_map="auto",
            local_files_only=True # prevent downloading from hub
        )
        logger.info("Model loaded")
        model.eval()  # Set to evaluation mode

        #summary(model)

        print("\n" + "="*70)
        print("EXPLORING Mistral3Model ATTRIBUTES")
        print("="*70)

        # Type and class
        print(f"\nType: {type(model.model)}")
        print(f"Class name: {model.model.__class__.__name__}")

        # List all attributes and methods of the model
        print("\nAttributes and methods of model.model:")
        for attr in dir(model.model):
            print(f"  {attr}")

    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
        sys.exit(1)
    
    logger.info("Script finished successfully!")

if __name__ == "__main__":
    main()