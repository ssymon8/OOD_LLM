import torch
import torch.nn as nn
from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend

def inspect_layer_hook(layer_idx):
    """
    Hook to inspect the inputs and outputs of a specific layer during the forward pass.
    """
    def hook(module, input, output):
        print(f"\n{'='*70}")
        print(f"Layer {layer_idx} - {module.__class__.__name__}")
        print(f"{'='*70}")
        
        # inspect inputs
        # print the Tenspr shape if it's a tensor, otherwise print the type
        if isinstance(input, tuple):
            print(f"Input (tuple): {len(input)} elements")
            for i, inp in enumerate(input):
                if isinstance(inp, torch.Tensor):
                    print(f"  [{i}] Tensor - shape: {inp.shape}")
                else:
                    print(f"  [{i}] {type(inp).__name__}")
        elif isinstance(input, torch.Tensor):
            print(f"Input (Tensor): shape: {input.shape}")

        ###########################################################
        #inspect the layers (self-attention, feedforward, etc.)
        if hasattr(module, "self_attn"):
            print("\nSelf-Attention Layer:")
            print(f"  Query shape: {module.self_attn.q_proj.weight.shape}")
            print(f"  Key shape: {module.self_attn.k_proj.weight.shape}")
            print(f"  Value shape: {module.self_attn.v_proj.weight.shape}")
        
        if hasattr(module, "mlp"):
            print("\nFeedforward Layer:")
            print(f"  Linear1 shape: {module.mlp.linear1.weight.shape}")
            print(f"  Linear2 shape: {module.mlp.linear2.weight.shape}")
        ###########################################################
        
        # inspect outputs
        if isinstance(output, tuple):
            print(f"output (tuple): {len(output)} elements")
            for o, out in enumerate(output):
                if isinstance(out, torch.Tensor):
                    print(f"  [{o}] Tensor - shape: {out.shape}")
                else:
                    print(f"  [{o}] {type(out).__name__}")
        elif isinstance(output, torch.Tensor):
            print(f"output (Tensor): shape: {output.shape}")
        
        print(f"{'='*70}\n")
    return hook