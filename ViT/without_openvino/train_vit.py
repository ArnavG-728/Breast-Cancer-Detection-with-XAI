
import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from torchvision.datasets.folder import default_loader
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths
DATASET_PATH = "augmented_dataset"
MODEL_SAVE_DIR = "ViT/models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Custom dataset class
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
        label = self.labels[idx]
        return image, label

# Load data from folder and assign labels based on subfolder names
def load_dataset(dataset_path):
    classes = os.listdir(dataset_path)
    image_paths, labels = [], []
    for idx, cls in enumerate(classes):
        cls_folder = os.path.join(dataset_path, cls)
        for img_name in os.listdir(cls_folder):
            image_paths.append(os.path.join(cls_folder, img_name))
            labels.append(idx)
    return image_paths, labels, classes

# Function to evaluate model
def evaluate_model(model, val_loader, device, class_names):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    class_report = classification_report(all_labels, all_preds, target_names=class_names)

    return accuracy, precision, recall, f1, cm, class_report

if __name__ == "__main__":
    # Load dataset paths and labels
    image_paths, labels, class_names = load_dataset(DATASET_PATH)

    # Train-validation split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # ViT image processor for normalization
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
    ])

    # Datasets and DataLoaders
    train_dataset = AugmentedDataset(train_paths, train_labels, transform)
    val_dataset = AugmentedDataset(val_paths, val_labels, transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

    # Load model with correct num_labels
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=len(set(labels)),
        ignore_mismatched_sizes=True
    ).to(device)

    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

    # Training loop
    epochs = 5
    print("\n🔥 Training Started...")
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            loop.set_postfix(loss=running_loss / total, acc=correct / total)

        # Save model for this epoch
        model_save_path = os.path.join(MODEL_SAVE_DIR, f"vit_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), model_save_path)
        print(f"💾 Model saved to {model_save_path}")

        # Evaluation
        val_accuracy, val_precision, val_recall, val_f1, val_cm, _ = evaluate_model(
            model, val_loader, device, class_names
        )

        print(f"\n📊 Epoch {epoch+1} Validation Metrics:")
        print(f"Accuracy : {val_accuracy:.4f}")
        print(f"Precision: {val_precision:.4f}")
        print(f"Recall   : {val_recall:.4f}")
        print(f"F1 Score : {val_f1:.4f}")
        print(f"Confusion Matrix:\n{val_cm}")

    # Final Evaluation
    print("\n🔍 Final Evaluation:")
    final_accuracy, final_precision, final_recall, final_f1, final_cm, final_report = evaluate_model(
        model, val_loader, device, class_names
    )

    print(f"\n✅ Final Accuracy   : {final_accuracy:.4f}")
    print(f"✅ Final Precision  : {final_precision:.4f}")
    print(f"✅ Final Recall     : {final_recall:.4f}")
    print(f"✅ Final F1 Score   : {final_f1:.4f}")
    print(f"🧠 Final Confusion Matrix:\n{final_cm}")
    print(f"\n📋 Classification Report:\n{final_report}")
    print("\n🏁 Training & Evaluation Complete!")
