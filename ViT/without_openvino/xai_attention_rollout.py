import os
import torch
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from transformers import ViTImageProcessor, ViTForImageClassification
from torchvision.datasets.folder import default_loader

# Rollout utility
def compute_rollout_attention(all_attentions):
    result = torch.eye(all_attentions[0].size(-1))
    for attention in all_attentions:
        attention_heads_fused = attention.mean(dim=1)
        attention_heads_fused = attention_heads_fused + torch.eye(attention_heads_fused.size(-1))
        attention_heads_fused = attention_heads_fused / attention_heads_fused.sum(dim=-1, keepdim=True)
        result = torch.matmul(attention_heads_fused, result)
    mask = result[0, 0, 1:]  # Exclude CLS token
    return mask

def visualize_rollout(attn_mask, image_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))

    heatmap = attn_mask.reshape(14, 14)
    heatmap = cv2.resize(heatmap.numpy(), (224, 224))
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    filename = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(save_dir, f"{filename}-rollout.png")
    cv2.imwrite(output_path, overlay)
    print(f"✅ Attention rollout heatmap saved to {output_path}")

if __name__ == "__main__":
    model_path = "ViT/models/vit_epoch_5.pt"
    image_path = "augmented_dataset/benign/benign (2)_aug1.png"
    save_dir = "ViT/xai_rollout"

    # Load model
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=3,
        ignore_mismatched_sizes=True
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # Load image
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
    ])

    image = default_loader(image_path)
    input_tensor = transform(image).unsqueeze(0)  # [1, 3, 224, 224]

    # Forward with attention output
    with torch.no_grad():
        outputs = model(input_tensor, output_attentions=True)
        attentions = outputs.attentions  # Tuple of layers [batch, heads, tokens, tokens]

    rollout_mask = compute_rollout_attention(attentions)
    visualize_rollout(rollout_mask, image_path, save_dir)
