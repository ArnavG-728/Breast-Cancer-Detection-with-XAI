import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from tqdm import tqdm
import numpy as np
from openvino.runtime import Core

# ===============================
# Setup
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0)
print(f"✅ Using device: {device}")

# Paths
DATASET_PATH = "augmented_dataset"
MODEL_SAVE_DIR = "ViT/models"
ONNX_PATH = os.path.join(MODEL_SAVE_DIR, "vit_model.onnx")
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# ===============================
# Dataset Definition
# ===============================
class AugmentedDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = default_loader(self.image_paths[idx])
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

# ===============================
# Dataset Loader
# ===============================
def load_dataset(dataset_path):
    classes = os.listdir(dataset_path)
    image_paths, labels = [], []
    for idx, cls in enumerate(classes):
        for img_file in os.listdir(os.path.join(dataset_path, cls)):
            image_paths.append(os.path.join(dataset_path, cls, img_file))
            labels.append(idx)
    return image_paths, labels, classes

# ===============================
# OpenVINO Evaluation
# ===============================
def evaluate_openvino(core, compiled_model, val_loader, input_key, output_key, class_names):
    all_labels, all_preds = [], []

    for images, labels in tqdm(val_loader, desc="🧠 OpenVINO Evaluation"):
        images = images.numpy()
        for img, label in zip(images, labels):
            input_tensor = np.expand_dims(img, axis=0)
            result = compiled_model({input_key: input_tensor})
            pred = np.argmax(result[output_key], axis=1)
            all_preds.append(pred[0])
            all_labels.append(label.item())

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names)
    return acc, precision, recall, f1, cm, report

# ===============================
# Main Script
# ===============================
if __name__ == "__main__":
    # Load dataset
    image_paths, labels, class_names = load_dataset(DATASET_PATH)

    # Train-validation split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, stratify=labels, random_state=42
    )

    # Processor and transform
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
    ])

    # Datasets and loaders
    train_dataset = AugmentedDataset(train_paths, train_labels, transform)
    val_dataset = AugmentedDataset(val_paths, val_labels, transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Load model
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=len(set(labels)),
        ignore_mismatched_sizes=True
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

    # ===============================
    # Training Loop
    # ===============================
    for epoch in range(5):  # 🔁 Change to range(5) for full training
        model.train()
        correct, total, total_loss = 0, 0, 0.0
        loop = tqdm(train_loader, desc=f"🔥 Epoch {epoch + 1}")

        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            loop.set_postfix(loss=total_loss / total, acc=correct / total)

        torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, f"vit_epoch_{epoch + 1}.pt"))

    # ===============================
    # Export to ONNX
    # ===============================
    print("📦 Exporting to ONNX...")
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,  # 🔁 Corrected to save at the same path being read
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"✅ ONNX model saved at: {ONNX_PATH}")

    # ===============================
    # OpenVINO Inference
    # ===============================
    print("🚀 OpenVINO Inference Starting...")
    core = Core()
    ov_model = core.read_model(ONNX_PATH)
    compiled_model = core.compile_model(ov_model, "CPU")

    input_key = compiled_model.input(0).get_any_name()
    output_key = compiled_model.output(0).get_any_name()

    acc, prec, rec, f1, cm, report = evaluate_openvino(
        core, compiled_model, val_loader, input_key, output_key, class_names
    )

    print("\n✅ Final Evaluation with OpenVINO:")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", report)
