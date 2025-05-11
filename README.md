# Vision Transformer Attention Visualization

This repository contains experiments for visualizing and analyzing attention mechanisms in Vision Transformers (ViT). Through attention visualization and rollout techniques, the project demonstrates how transformers "see" and process images, offering insights into their internal mechanisms.

## 📋 Overview

Vision Transformers have shown remarkable performance in computer vision tasks, but they can be difficult to interpret. This project implements techniques to visualize and analyze attention patterns in ViTs through three key experiments:

1. **Attention visualization for single-object images** - Visualizing how the model attends to different parts of an image containing a single main object
2. **Attention analysis for multi-object images** - Comparing raw attention vs. attention rollout visualization for images with multiple objects
3. **Object embedding similarity analysis** - Quantifying how attention-weighted representations differ from standard embeddings

## 🚀 Key Features

- Implementation of attention rollout algorithm for transformer interpretability
- Visualization of CLS token attention weights and their overlay on original images
- Entropy calculation for attention distributions
- Object detection and embedding extraction for multiple objects in an image
- Comparison of embedding similarities with and without attention rollout

## 🛠️ Installation

```bash
# Clone this repository
git clone https://github.com/yourusername/vit-attention-visualization.git
cd vit-attention-visualization

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install torch torchvision
pip install transformers
pip install timm
pip install pillow
pip install matplotlib seaborn
pip install requests tqdm
```

## 💡 Attention Rollout Explained

Attention rollout is a technique to visualize how information flows through all layers of a transformer, as described in [Abnar & Zuidema (2020)](https://arxiv.org/abs/2005.00928). The implementation:

1. Starts with the identity matrix (each token attends to itself)
2. For each layer, adds residual connections to the attention matrix and normalizes
3. Multiplies attention matrices cumulatively across layers
4. Results in a single matrix showing the integrated attention flow throughout the network

## 📊 Experiments

### Experiment 1: Single Object Attention Visualization

Visualizes how Vision Transformers attend to different parts of an image with a single main object:

```python
# Example of visualizing attention maps
attn_map = model.blocks[-1].attn.attn_map.mean(dim=1).squeeze(0).detach()
cls_weight = model.blocks[-1].attn.cls_attn_map.max(dim=1).values.view(14, 14).detach()
plot_set(image, attn_map_cpu, cls_weight_cpu, img_resized, cls_resized_cpu)
```

### Experiment 2: Multiple Object Attention Analysis

Compares raw attention from the final layer with attention rollout across all layers for images containing multiple objects:

```python
# Raw attention vs. Attention rollout visualization
fig, axes = plt.subplots(1, 2, figsize=(10,4))
axes[0].imshow(image)
axes[0].imshow(attn_map_raw_resized, cmap='jet', alpha=0.4)
axes[0].set_title("Raw Attention (last layer)")

axes[1].imshow(image)
axes[1].imshow(attn_map_rollout_resized, cmap='jet', alpha=0.4)
axes[1].set_title("Attention Rollout (all layers)")
```

### Experiment 3: Object Embedding Similarity

Analyzes how attention rollout affects the similarity between object embeddings:

```python
# Compare cosine similarities between standard and rollout-based embeddings
for i, label_i in enumerate(labels):
    for j, label_j in enumerate(labels):
        if j <= i:
            continue
        sim_std = cosine_sim(embeddings_std[label_i], embeddings_std[label_j])
        sim_roll = cosine_sim(embeddings_rollout[label_i], embeddings_rollout[label_j])
        print(f"{label_i} vs {label_j}: standard sim = {sim_std:.3f}, rollout sim = {sim_roll:.3f}")
```

## 📸 Sample Results

The experiments reveal several interesting findings:

- Attention maps often focus on relevant object parts and boundaries
- Attention rollout provides a more distributed view of attention across the image compared to raw attention
- Semantically related objects (e.g., dog and cat) generally have higher embedding similarity
- The attention rollout technique affects similarity scores between object embeddings, typically reducing similarity values

## 📚 References

1. Abnar, S. and Zuidema, W. (2020). [Quantifying Attention Flow in Transformers](https://arxiv.org/abs/2005.00928). Proceedings of EMNLP.
2. Dosovitskiy, A., et al. (2020). [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929). ICLR 2021.
3. Carion, N., et al. (2020). [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872). ECCV 2020.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
