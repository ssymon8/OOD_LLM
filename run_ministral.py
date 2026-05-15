import os
import sys
import torch
import json
from pathlib import Path
from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend, FineGrainedFP8Config

import logging

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
        # Tokenizer loading
        tokenizer = MistralCommonBackend.from_pretrained(model_path)
        logger.info("Tokenizer loaded")
        
        # Model loading
        model = Mistral3ForConditionalGeneration.from_pretrained(
            model_path,
            device_map="auto",
            local_files_only=True,  # prevent downloading from hub
            quantization_config=FineGrainedFP8Config(dequantize= True)
        )
        logger.info("Model loaded")
        model.eval()  # Set to evaluation mode

        # requests to test the model
        prompts = [
            "Explique le concept de pointeur intelligent (smart pointer) en C++.",
            "Écris une fonction Python utilisant PyTorch pour multiplier deux tenseurs."
        ]

        # Create output directory
        output_dir = Path("./outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
    
        output_file = output_dir / "reponses.json"

        with open(output_file, "w", encoding="utf-8") as f:
            for i, prompt in enumerate(prompts, 1):
                logger.info(f"Processing prompt {i}/{len(prompts)}: '{prompt[:30]}...'")
                messages = [{"role": "user", "content": prompt}]
                
                # Apply chat template and move all tensors to device
                inputs = tokenizer.apply_chat_template(
                    messages,
                    return_tensors="pt",
                    return_dict=True
                ).to(device)

                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask"),
                        max_length=inputs["input_ids"].shape[1] + 512,
                        temperature=0.1,
                        do_sample=True
                    )
                
                # Decode response, skipping input tokens
                input_length = inputs["input_ids"].shape[1]
                response = tokenizer.decode(
                    outputs[0][input_length:],
                    skip_special_tokens=True
                )
                
                # Save result
                json.dump({"prompt": prompt, "response": response}, f, ensure_ascii=False)
                f.write("\n")

        logger.info(f"Results saved to {output_file}")
    
    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()