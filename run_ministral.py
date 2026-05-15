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

    # Exemple de requêtes à traiter en lot
    prompts = [
        "Explique le concept de pointeur intelligent (smart pointer) en C++.",
        "Écris une fonction Python utilisant PyTorch pour multiplier deux tenseurs."
    ]

    with open(output_file, "w", encoding="utf-8") as f:
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            
            # Préparation du prompt avec le template du modèle
            inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
            
            print(f"Génération pour le prompt : '{prompt[:30]}...'")
            outputs = model.generate(
                inputs, 
                max_new_tokens=512,
                temperature=0.3, # Faible pour du code/technique
                do_sample=True
            )
            
            # Décodage en ignorant le prompt initial
            response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
            
            f.write(f"PROMPT:\n{prompt}\n\nREPONSE:\n{response}\n")
            f.write("="*50 + "\n")

    print(f"Terminé. Résultats sauvegardés dans {output_file}")

if __name__ == "__main__":
    main()