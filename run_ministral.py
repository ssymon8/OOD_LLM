import os
import sys
import torch
import json
from pathlib import Path
from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend, FineGrainedFP8Config

import logging

from extractor_utils import get_layer_output

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
        # env variable for hf cache
        os.environ["HF_HOME"] = "/root/.cache/huggingface"

        if not torch.cuda.is_available():
            logger.error("CUDA not available :/")
            sys.exit(1)
        
        device = torch.device("cuda")
        logger.info(f"Using device: {device}")

        MODEL_ID = "mistralai/Ministral-3-3B-Base-2512"

        logger.info(f"loading tokenizer for {MODEL_ID}...")
        tokenizer = MistralCommonBackend.from_pretrained(MODEL_ID)
        logger.info("Tokenizer loaded")

        logger.info(f"Loading model from {MODEL_ID}...")
        ministral = Mistral3ForConditionalGeneration.from_pretrained(
            MODEL_ID,
            device_map="auto",
            quantization_config=FineGrainedFP8Config(dequantize = True))
        logger.info("Model loaded")
        ministral.eval()  # Set to evaluation mode

        # register the hook on 25th layer of the model
        target_layer_idx = 25
        # register the hook on the target layer
        hook, features = get_layer_output(target_layer_idx)
        handle = ministral.model.layers[target_layer_idx].register_forward_hook(hook)
        logger.info(f"Registered hook on layer {target_layer_idx}")
        
        #list of the embedded outputs of the 25th layer
        layer_outputs = []
        # requests to test the model
        prompts = [
            "Explique le concept de pointeur intelligent (smart pointer) en C++.",
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
                    outputs = ministral.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask"),
                        max_length=inputs["input_ids"].shape[1] + 512,
                        temperature=0.1,
                        do_sample=True
                    )

                #append the layer output to the list
                if "outputs" in features:
                    layer_outputs.append(features["outputs"])
                    logger.info(f"Captured output from layer {target_layer_idx} with shape: {features['outputs'].shape}")

                handle.remove()  # Remove the hook after use

                
                # Decode response, skipping input tokens
                input_length = inputs["input_ids"].shape[1]
                response = tokenizer.decode(
                    outputs[0][input_length:],
                    skip_special_tokens=True
                )
                
                # Save result
                json.dump({"prompt": prompt, "response": response}, f, ensure_ascii=False)
                f.write("\n")
            
            # Save the captured layer outputs as well
            layer_outputs_file = output_dir / "layer_outputs.pt"
            torch.save(layer_outputs, layer_outputs_file)

        logger.info(f"Results saved to {output_file}")
    
    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()