import os
import torch
import json
from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend

def main():
    model_path = os.path.expanduser("~/OOD_LLM/ministral")
    
    print(f"Loading model from {model_path}...")
    
    # Tokenizer loading
    tokenizer = MistralCommonBackend.from_pretrained(model_path)
    
    model = Mistral3ForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.fp8, #model is in fp8, so we load it directly in fp8 for efficiency
        device_map="auto",
        local_files_only=True # additional safeguard to ensure we only load from local files
    )

    # requests to test the model
    prompts = [
        "Explique le concept de pointeur intelligent (smart pointer) en C++.",
        "Écris une fonction Python utilisant PyTorch pour multiplier deux tenseurs."
    ]

    if not os.path.exists("./outputs/reponses.txt"):
        os.makedirs("outputs")
    
    output_file = "./outputs/reponses.txt"

    with open(output_file, "w", encoding="utf-8") as f:
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            
            # prompt processing using the tokenizer's chat template
            inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
            
            print(f"generating response for prompt : '{prompt[:30]}...'")
            outputs = model.generate(
                inputs, 
                max_new_tokens=512,
                temperature=0.1, # respecting the hf readme
                do_sample=True
            )
            
            # decode the generated response, skipping the input tokens
            response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
            
            f.write(f"PROMPT:\n{prompt}\n\nREPONSE:\n{response}\n")
            f.write("="*50 + "\n")

    print(f"Done, results saved to {output_file}")

if __name__ == "__main__":
    main()