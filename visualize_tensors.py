import torch
import argparse
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
import sys
import os

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
    parser.add_argument("--folder", type=str, default = "./outputs/sociology", help="Path to the folder containing the saved layer outputs files")
    parser.add_argument("--mode", type= str, choices = ["extract", "visualize"], default = "extract", help = "Select the mode of the script" )
    args = parser.parse_args()

    assert args.mode is not None

    if len(args.folder.strip()) == 0:
        print("Error, empty folder.", file=sys.stderr)
        sys.exit(1)

    if not os.path.isdir(args.folder):
        print(f"Error : folder not valid.", file=sys.stderr)
        sys.exit(1)

    if args.mode == "extract":
        Path(f"{args.folder}/processed").mkdir(parents= True, exist_ok= True)
        for layer_idx in range(26):
            correct_file = Path(args.folder)/ f"correct_outputs_layer_{layer_idx}.pt"
            wrong_file = Path(args.folder)/ f"wrong_outputs_layer_{layer_idx}.pt"

            if correct_file.exists():
                correct_data = torch.load(correct_file, map_location = 'cpu')
                print(f"Type of first element: {type(correct_data[0])}")
                last_token_correct = extract_last_token_from_layer_outputs(correct_data)
                correct_path = Path(f"{args.folder}/processed/correct_last_token_layer_{layer_idx}.pt")
                torch.save(last_token_correct, correct_path)
            if wrong_file.exists():
                wrong_data = torch.load(wrong_file, map_location = 'cpu')
                print(f"Type of first element: {type(wrong_data[0])}")
                last_token_wrong = extract_last_token_from_layer_outputs(wrong_data)
                wrong_path = Path(f"{args.folder}/processed/wrong_last_token_layer_{layer_idx}.pt")
                torch.save(last_token_wrong, wrong_path)

            del correct_data, wrong_data        

    """
    Token ID for choice 'A': 1065
    Token ID for choice 'B': 1066
    Token ID for choice 'C': 1067
    Token ID for choice 'D': 1068
    """

    if args.mode == "visualize":
        Path(f"./visuals").mkdir(parents = True, exist_ok = True)
        images= []
        
        pca = PCA(n_components=50)
        umap = UMAP(n_components=3, n_neighbors=15, min_dist=0.1, random_state=42)

        for layer_idx in range(26):
            correct_path = Path(f"{args.folder}/processed/correct_last_token_layer_{layer_idx}.pt")
            wrong_path = Path(f"{args.folder}/processed/wrong_last_token_layer_{layer_idx}.pt")
            
                

            if correct_path.exists() and wrong_path.exists():
                last_token_correct = torch.load(correct_path, map_location= 'cpu')
                last_token_wrong = torch.load(wrong_path, map_location = 'cpu')


                # Perform PCA on the last tokens
                all_outputs = last_token_correct + last_token_wrong
                labels = np.concatenate([np.ones(len(last_token_correct)), np.zeros(len(last_token_wrong))])


                stack_tokens = torch.stack(all_outputs, dim=0).to(torch.float32).numpy()

                tokens_pca = pca.fit_transform(stack_tokens.reshape(stack_tokens.shape[0],-1))
                tokens_3d_umap = umap.fit_transform(tokens_pca)

                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(tokens_3d_umap[labels==1, 0], tokens_3d_umap[labels==1, 1], tokens_3d_umap[labels==1, 2], 
                   alpha=0.7, c='green', s=50)
                ax.scatter(tokens_3d_umap[labels==0, 0], tokens_3d_umap[labels==0, 1], tokens_3d_umap[labels==0, 2], 
                   alpha=0.7, c='red', s=50)
                
                ax.set_xlim([-3, 13])
                ax.set_ylim([-3, 13])
                ax.set_zlim([-3, 13]) 
        
                plt.savefig(f'./visuals/frame_layer_{layer_idx}.png', dpi=80)
                images.append(Image.open(f'./visuals/frame_layer_{layer_idx}.png'))
                plt.close()

        images[0].save('umap_3d_layerwise.gif', save_all=True, append_images=images[1:], duration=100, loop=0)
    

        

    """
    plt.figure(figsize=(10, 7))
    plt.scatter(tokens_3d_umap[labels==1, 0], tokens_3d_umap[labels==1, 1], alpha=0.7, label='Correct', c='green')
    plt.scatter(tokens_3d_umap[labels==0, 0], tokens_3d_umap[labels==0, 1], alpha=0.7, label='Wrong', c='red')
    plt.savefig("umap_viz.png")

    plt.figure(figsize=(10, 7))
    plt.scatter(tokens_2d[labels==1, 0], tokens_2d[labels==1, 1], alpha=0.7, label='Correct', c='green')
    plt.scatter(tokens_2d[labels==0, 0], tokens_2d[labels==0, 1], alpha=0.7, label='Wrong', c='red')
    plt.legend()
    plt.title(f"t-SNE - college chemistry")
    plt.savefig("t_SNE_viz.png")
    
    images = []
    for angle in range(0, 360, 10):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(tokens_3d_umap[labels==1, 0], tokens_3d_umap[labels==1, 1], tokens_3d_umap[labels==1, 2], 
                   alpha=0.7, c='green', s=50)
        ax.scatter(tokens_3d_umap[labels==0, 0], tokens_3d_umap[labels==0, 1], tokens_3d_umap[labels==0, 2], 
                   alpha=0.7, c='red', s=50)
        ax.view_init(elev=20, azim=angle)
        
        plt.savefig(f'frame_{angle}.png', dpi=80)
        images.append(Image.open(f'frame_{angle}.png'))
        plt.close()
    
    images[0].save('umap_3d.gif', save_all=True, append_images=images[1:], duration=100, loop=0)
"""

if __name__ == "__main__":
    main()