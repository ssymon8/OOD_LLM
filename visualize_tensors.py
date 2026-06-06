import torch

data = torch.load("layer_outputs.pt", map_location="cpu")

print(f"Structure type: {type(data)}")
print(f"Number of elements in the list: {len(data)}")

if len(data) > 0:
    first_element = data[0]
    print(f"Type of element 0: {type(first_element)}")
    if isinstance(first_element, torch.Tensor):
        print(f"Shape of element 0: {first_element.shape}")
        print(f"Dtype of element 0: {first_element.dtype}")
    elif isinstance(first_element, dict):
        print(f"Keys of the dictionary: {list(first_element.keys())}")