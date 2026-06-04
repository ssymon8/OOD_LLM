import os
import sys
import torch
import json
from pathlib import Path
from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend, FineGrainedFP8Config
import logging


def infer_with_clean_context(prompt: str, model, tokenizer):
    """
    Assuming the model and tokenizer are already loaded, this
    function aims at infering on the given prompt by cleaning the 
    context and removing any special tokens that might be present in the input.

    Args:
        prompt (str): The input prompt for inference.
        model: The loaded model for inference.
        tokenizer: The loaded tokenizer for inference.

    Returns:
        output (str): The generated output (with 0 context)
    """

    # Cleaning the context first.
    torch.cuda.empty_cache()  # Clear GPU memory before inference

    # Tokenize the prompt
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(messages,return_tensors="pt",return_dict=True).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_length=inputs["input_ids"].shape[1] + 512,
            temperature=0.1,
            do_sample=True
        )

    # Decode the output, skipping input tokens
    input_length = inputs["input_ids"].shape[1]

    response = tokenizer.decode(
        outputs[0][input_length:],
        skip_special_tokens=True
    )

    del inputs, outputs  # Free up memory
    torch.cuda.empty_cache()  # Clear GPU memory after inference, just in case

    return response

def mmlu_five_shot_prompt(prompt: str, model, tokenizer):
    """
    Similar to the above function but specifically designed for MMLU 5-shot inference.
    It will format the prompt according to the MMLU 5-shot format and then perform inference.
    """
    