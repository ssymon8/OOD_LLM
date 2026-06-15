import torch
import argparse
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

import imageio
from PIL import Image

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def extract_last_token_from_layer_outputs(layer_outputs):
    last_tokens = [layer_output[:, -1, :] for layer_output in layer_outputs]
    return last_tokens

def main():
    
    parser = argparse.ArgumentParser(description="Visualize captured layer outputs")
    parser.add_argument("--file", type=str, default="layer_outputs.pt", help="Path to the saved layer outputs file")
    args = parser.parse_args()

    correct_file = f"./outputs/college_chemistry_correct.pt"
    wrong_file = f"./outputs/college_chemistry_wrong.pt"

    correct_outputs = torch.load(correct_file, map_location = 'cpu')
    wrong_outputs = torch.load(wrong_file, map_location = 'cpu')

    """
    print(f"Structure type: {type(data)}")
    print(f"Number of elements in the list: {len(data)}")

    if len(data) > 0:
        first_element = data[1]
        print(f"Type of element 0: {type(first_element)}")
        if isinstance(first_element, torch.Tensor):
            print(f"Shape of element 0: {first_element.shape}")
            print(f"Dtype of element 0: {first_element.dtype}")
        elif isinstance(first_element, dict):
            print(f"Keys of the dictionary: {list(first_element.keys())}")
    
    print(data[1])
    Token ID for choice 'A': 1065
    Token ID for choice 'B': 1066
    Token ID for choice 'C': 1067
    Token ID for choice 'D': 1068
    
    last_tokens= extract_last_token_from_layer_outputs(data)
    print(f"Extracted last tokens from {len(last_tokens)} layer outputs.")

    file_path = Path(f"./outputs/college_chemistry_wrong.pt")
    torch.save(last_tokens, file_path)

    """
    # Perform PCA on the last tokens
    all_outputs = correct_outputs + wrong_outputs
    labels = np.concatenate([np.ones(len(correct_outputs)), np.zeros(len(wrong_outputs))])


    stack_tokens = torch.stack(all_outputs, dim=0).to(torch.float32).numpy()

    pca = PCA(n_components=50)
    tokens_pca = pca.fit_transform(stack_tokens.reshape(stack_tokens.shape[0],-1))

    tsne = TSNE(n_components =2, random_state = 42, perplexity = 10)
    tokens_2d = tsne.fit_transform(tokens_pca)

    
    umap = UMAP(n_components=3, n_neighbors=15, min_dist=0.1, random_state=42)
    tokens_2d_umap = umap.fit_transform(tokens_pca)
    """
    plt.figure(figsize=(10, 7))
    plt.scatter(tokens_2d_umap[labels==1, 0], tokens_2d_umap[labels==1, 1], alpha=0.7, label='Correct', c='green')
    plt.scatter(tokens_2d_umap[labels==0, 0], tokens_2d_umap[labels==0, 1], alpha=0.7, label='Wrong', c='red')
    plt.savefig("umap_viz.png")

    plt.figure(figsize=(10, 7))
    plt.scatter(tokens_2d[labels==1, 0], tokens_2d[labels==1, 1], alpha=0.7, label='Correct', c='green')
    plt.scatter(tokens_2d[labels==0, 0], tokens_2d[labels==0, 1], alpha=0.7, label='Wrong', c='red')
    plt.legend()
    plt.title(f"t-SNE - college chemistry")
    plt.savefig("t_SNE_viz.png")
    
    """
        
    
    images = []
    for angle in range(0, 360, 10):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(tokens_2d_umap[labels==1, 0], tokens_2d_umap[labels==1, 1], tokens_2d_umap[labels==1, 2], 
                   alpha=0.7, c='green', s=50)
        ax.scatter(tokens_2d_umap[labels==0, 0], tokens_2d_umap[labels==0, 1], tokens_2d_umap[labels==0, 2], 
                   alpha=0.7, c='red', s=50)
        ax.view_init(elev=20, azim=angle)
        
        plt.savefig(f'frame_{angle}.png', dpi=80)
        images.append(Image.open(f'frame_{angle}.png'))
        plt.close()
    
    images[0].save('umap_3d.gif', save_all=True, append_images=images[1:], duration=100, loop=0)

if __name__ == "__main__":
    main()