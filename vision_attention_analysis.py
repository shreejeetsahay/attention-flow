import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, ToTensor
from timm.models import create_model
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from PIL import Image
from tqdm.auto import tqdm
import os
import networkx as nx
from typing import List, Tuple, Dict, Optional, Union, Callable
import warnings
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")


class TransformerAttentionAnalyzer:
    def __init__(self, model_name: str = 'deit_small_distilled_patch16_224', pretrained: bool = True):
        self.model = create_model(model_name, pretrained=pretrained).to(device)
        self.patch_size = 16  
        self.img_size = 224   
        self.grid_size = self.img_size // self.patch_size
        self.register_attention_hooks()
        
    def register_attention_hooks(self):
        for i, block in enumerate(self.model.blocks):
            block.attn.forward = self._make_forward_hook(block.attn, i)
            
    def _make_forward_hook(self, attn_obj, block_idx):
        def forward_hook(x):
            B, N, C = x.shape
            qkv = attn_obj.qkv(x).reshape(B, N, 3, attn_obj.num_heads, C // attn_obj.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)

            attn = (q @ k.transpose(-2, -1)) * attn_obj.scale
            attn = attn.softmax(dim=-1)
            attn = attn_obj.attn_drop(attn)
            
            attn_obj.attn_map = attn
            attn_obj.cls_attn_map = attn[:, :, 0, 2:]

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = attn_obj.proj(x)
            x = attn_obj.proj_drop(x)
            return x
        
        return forward_hook
        
    def preprocess_image(self, image):
        transform = Compose([
            Resize(249, 3),
            CenterCrop(224),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        return transform(image).unsqueeze(0).to(device)
    
    def analyze_image(self, image):
        if isinstance(image, str) and image.startswith(('http://', 'https://')):
            image = Image.open(requests.get(image, stream=True).raw)
        
        x = self.preprocess_image(image)
        output = self.model(x)
        
        attn_maps = []
        cls_weights = []
        
        for block in self.model.blocks:
            attn_maps.append(block.attn.attn_map.detach())
            cls_weights.append(block.attn.cls_attn_map.detach())
            
        return output, attn_maps, cls_weights
    
    def compute_attention_rollout(self, attn_maps, include_residual=True):
        attn_maps_cpu = [attn.cpu() for attn in attn_maps]
        
        B, H, N, N = attn_maps_cpu[0].shape
        
        attn_maps_mean = [attn_map.mean(dim=1) for attn_map in attn_maps_cpu]
        
        if include_residual:
            attn_maps_mean = [0.5 * attn_map + 0.5 * torch.eye(N) for attn_map in attn_maps_mean]
        
        attention_rollout = []
        eye = torch.eye(N)
        
        for i, attn_map in enumerate(attn_maps_mean):
            if i == 0:
                attention_rollout.append(attn_map)
            else:
                rolled = attention_rollout[-1] @ attn_map
                rolled = rolled / rolled.sum(dim=-1, keepdim=True)
                attention_rollout.append(rolled)
                
        return attention_rollout
    
    def compute_attention_flow(self, attn_maps, include_residual=True):
        attn_maps_cpu = [attn.cpu() for attn in attn_maps]
        
        B, H, N, N = attn_maps_cpu[0].shape
        
        attn_maps_mean = [attn_map.mean(dim=1).squeeze(0) for attn_map in attn_maps_cpu]
        
        if include_residual:
            attn_maps_mean = [0.5 * attn_map + 0.5 * torch.eye(N) for attn_map in attn_maps_mean]
        
        attention_flows = []
        
        attention_flows.append(attn_maps_mean[0])
        
        for layer_idx in range(1, len(attn_maps_mean)):
            G = nx.DiGraph()
            
            for l in range(layer_idx + 1):
                for i in range(N):
                    G.add_node(f"L{l}_N{i}")
            
            for i in range(N):
                G.add_edge("source", f"L{layer_idx}_N{i}", capacity=1.0)
            
            for i in range(N):
                G.add_edge(f"L0_N{i}", "sink", capacity=1.0)
            
            for l in range(layer_idx, 0, -1):
                current_attn = attn_maps_mean[l-1]
                for i in range(N):
                    for j in range(N):
                        weight = float(current_attn[i, j].item())
                        if weight > 0:
                            G.add_edge(f"L{l}_N{i}", f"L{l-1}_N{j}", capacity=weight)
            
            flow_matrix = torch.zeros((N, N))
            
            for src_idx in range(N):
                src_node = f"L{layer_idx}_N{src_idx}"
                
                for tgt_idx in range(N):
                    tgt_node = f"L0_N{tgt_idx}"
                    
                    H = G.copy()
                    
                    for i in range(N):
                        if i != src_idx:
                            H.remove_edge("source", f"L{layer_idx}_N{i}")
                    
                    for i in range(N):
                        if i != tgt_idx:
                            H.remove_edge(f"L0_N{i}", "sink")
                    
                    flow_value, _ = nx.maximum_flow(H, "source", "sink")
                    flow_matrix[src_idx, tgt_idx] = flow_value
            
            flow_matrix = flow_matrix / (flow_matrix.sum(dim=-1, keepdim=True) + 1e-10)
            attention_flows.append(flow_matrix)
            
        return attention_flows
    
    def visualize_attention(self, image, attn_rollout, attn_flow, layer_idx=-1, cls_idx=0):
        img_np = np.array(image)
        
        rollout_map = attn_rollout[layer_idx][cls_idx, 1:].reshape(self.grid_size, self.grid_size)
        flow_map = attn_flow[layer_idx][cls_idx, 1:].reshape(self.grid_size, self.grid_size)
        
        rollout_resized = F.interpolate(
            rollout_map.view(1, 1, self.grid_size, self.grid_size), 
            size=(self.img_size, self.img_size), 
            mode='bilinear'
        ).squeeze().numpy()
        
        flow_resized = F.interpolate(
            flow_map.view(1, 1, self.grid_size, self.grid_size), 
            size=(self.img_size, self.img_size), 
            mode='bilinear'
        ).squeeze().numpy()
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        axes[0].imshow(img_np)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        axes[1].imshow(img_np)
        im1 = axes[1].imshow(rollout_resized, alpha=0.7, cmap='jet')
        axes[1].set_title(f"Attention Rollout (Layer {layer_idx})")
        axes[1].axis('off')
        
        axes[2].imshow(img_np)
        im2 = axes[2].imshow(flow_resized, alpha=0.7, cmap='jet')
        axes[2].set_title(f"Attention Flow (Layer {layer_idx})")
        axes[2].axis('off')
        
        fig.colorbar(im1, ax=axes[1])
        fig.colorbar(im2, ax=axes[2])
        
        plt.tight_layout()
        plt.show()
        
    def visualize_all_layers(self, image, attn_rollout, attn_flow, cls_idx=0, num_cols=4):
        num_layers = len(attn_rollout)
        num_rows = (num_layers + num_cols - 1) // num_cols
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(4*num_cols, 4*num_rows))
        
        img_np = np.array(image)
        
        for layer_idx in range(num_layers):
            row = layer_idx // num_cols
            col = layer_idx % num_cols
            
            if num_rows == 1:
                ax = axes[col]
            else:
                ax = axes[row, col]
            
            rollout_map = attn_rollout[layer_idx][cls_idx, 1:].reshape(self.grid_size, self.grid_size)
            
            rollout_resized = F.interpolate(
                rollout_map.view(1, 1, self.grid_size, self.grid_size), 
                size=(self.img_size, self.img_size), 
                mode='bilinear'
            ).squeeze().numpy()
            
            ax.imshow(img_np)
            im = ax.imshow(rollout_resized, alpha=0.7, cmap='jet')
            ax.set_title(f"Layer {layer_idx}")
            ax.axis('off')
        
        for i in range(num_layers, num_rows * num_cols):
            row = i // num_cols
            col = i % num_cols
            if num_rows == 1:
                ax = axes[col]
            else:
                ax = axes[row, col]
            ax.axis('off')
        
        plt.suptitle("Attention Rollout Across Layers", fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(4*num_cols, 4*num_rows))
        
        for layer_idx in range(num_layers):
            row = layer_idx // num_cols
            col = layer_idx % num_cols
            
            if num_rows == 1:
                ax = axes[col]
            else:
                ax = axes[row, col]
            
            flow_map = attn_flow[layer_idx][cls_idx, 1:].reshape(self.grid_size, self.grid_size)
            
            flow_resized = F.interpolate(
                flow_map.view(1, 1, self.grid_size, self.grid_size), 
                size=(self.img_size, self.img_size), 
                mode='bilinear'
            ).squeeze().numpy()
            
            ax.imshow(img_np)
            im = ax.imshow(flow_resized, alpha=0.7, cmap='jet')
            ax.set_title(f"Layer {layer_idx}")
            ax.axis('off')
        
        for i in range(num_layers, num_rows * num_cols):
            row = i // num_cols
            col = i % num_cols
            if num_rows == 1:
                ax = axes[col]
            else:
                ax = axes[row, col]
            ax.axis('off')
        
        plt.suptitle("Attention Flow Across Layers", fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()
    
    def compare_attention_methods(self, rollout_attentions, flow_attentions):
        divergences = []
        
        for layer_idx in range(len(rollout_attentions)):
            rollout_attn = rollout_attentions[layer_idx][0, 1:]
            flow_attn = flow_attentions[layer_idx][0, 1:]
            
            rollout_attn = rollout_attn / rollout_attn.sum()
            flow_attn = flow_attn / flow_attn.sum()
            
            js_div = jensenshannon(rollout_attn.numpy(), flow_attn.numpy())
            divergences.append(js_div)
        
        return divergences
    
    def analyze_residual_impact(self, attn_maps, include_residuals=[True, False]):
        results = {}
        
        for include_residual in include_residuals:
            suffix = "with_residual" if include_residual else "without_residual"
            
            rollout_attns = self.compute_attention_rollout(attn_maps, include_residual=include_residual)
            
            entropies = []
            for layer_idx in range(len(rollout_attns)):
                attn_dist = rollout_attns[layer_idx][0, 1:]
                attn_dist = attn_dist / attn_dist.sum()
                
                entropy = -torch.sum(attn_dist * torch.log2(attn_dist + 1e-10))
                entropies.append(entropy.item())
            
            results[suffix] = {
                "entropies": entropies,
                "rollout_attns": rollout_attns
            }
        
        return results
    
    def visualize_residual_impact(self, residual_analysis):
        with_residual = residual_analysis["with_residual"]["entropies"]
        without_residual = residual_analysis["without_residual"]["entropies"]
        
        layers = list(range(len(with_residual)))
        
        plt.figure(figsize=(10, 6))
        plt.plot(layers, with_residual, 'o-', label='With Residual Connections')
        plt.plot(layers, without_residual, 's-', label='Without Residual Connections')
        plt.xlabel('Layer')
        plt.ylabel('Attention Entropy (bits)')
        plt.title('Impact of Residual Connections on Attention Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def compare_attn_to_patch_distance(self, image, attn_rollout, attn_flow, layer_idx=-1):
        rollout_attn = attn_rollout[layer_idx][0, 1:].reshape(self.grid_size, self.grid_size)
        flow_attn = attn_flow[layer_idx][0, 1:].reshape(self.grid_size, self.grid_size)
        
        center_i, center_j = self.grid_size // 2, self.grid_size // 2
        i_coords, j_coords = np.meshgrid(np.arange(self.grid_size), np.arange(self.grid_size))
        distances = np.sqrt((i_coords - center_i)**2 + (j_coords - center_j)**2)
        
        flat_distances = distances.flatten()
        flat_rollout = rollout_attn.flatten().numpy()
        flat_flow = flow_attn.flatten().numpy()
        
        rollout_corr, _ = pearsonr(flat_distances, flat_rollout)
        flow_corr, _ = pearsonr(flat_distances, flat_flow)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        im0 = axes[0].imshow(distances, cmap='viridis')
        axes[0].set_title("Distance from Center")
        fig.colorbar(im0, ax=axes[0])
        axes[0].axis('off')
        
        im1 = axes[1].imshow(rollout_attn, cmap='jet')
        axes[1].set_title(f"Attention Rollout (corr={rollout_corr:.3f})")
        fig.colorbar(im1, ax=axes[1])
        axes[1].axis('off')
        
        im2 = axes[2].imshow(flow_attn, cmap='jet')
        axes[2].set_title(f"Attention Flow (corr={flow_corr:.3f})")
        fig.colorbar(im2, ax=axes[2])
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return rollout_corr, flow_corr

def run_analysis():
    analyzer = TransformerAttentionAnalyzer()
    
    url = 'https://media.istockphoto.com/id/155439315/photo/passenger-airplane-flying-above-clouds-during-sunset.jpg?s=612x612&w=0&k=20&c=LJWadbs3B-jSGJBVy9s0f8gZMHi2NvWFXa3VJ2lFcL0='
    image = Image.open(requests.get(url, stream=True).raw)
    
    output, attn_maps, cls_weights = analyzer.analyze_image(image)
    
    attn_rollout = analyzer.compute_attention_rollout(attn_maps)
    attn_flow = analyzer.compute_attention_flow(attn_maps)
    
    analyzer.visualize_attention(image, attn_rollout, attn_flow)
    
    analyzer.visualize_all_layers(image, attn_rollout, attn_flow)
    
    divergences = analyzer.compare_attention_methods(attn_rollout, attn_flow)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(divergences)), divergences, 'o-')
    plt.xlabel('Layer')
    plt.ylabel('Jensen-Shannon Divergence')
    plt.title('Difference Between Attention Rollout and Flow Methods')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    residual_results = analyzer.analyze_residual_impact(attn_maps)
    analyzer.visualize_residual_impact(residual_results)
    
    analyzer.compare_attn_to_patch_distance(image, attn_rollout, attn_flow)
    
    return {
        'model': analyzer.model,
        'image': image,
        'attn_maps': attn_maps,
        'attn_rollout': attn_rollout,
        'attn_flow': attn_flow,
        'divergences': divergences,
        'residual_results': residual_results
    }

if __name__ == "__main__":
    results = run_analysis()