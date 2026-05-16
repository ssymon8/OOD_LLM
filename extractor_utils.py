import torch
import torch.nn as nn
from tranformers import Mistral3ForConditionalGeneration, MistralCommonBackend

def debug_hook(module, input, output):
    print(f"Debug Hook - Module: {module.__class__.__name__}")
    print(f"Input shape: {[i.shape for i in input]}")
    print(f"Output shape: {[o.shape for o in output]}")