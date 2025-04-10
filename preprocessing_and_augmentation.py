import os
import random
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.utils import save_image
import torch

# Set output size
resize_shape = (224, 224)

# Directories
input_root = "raw_dataset"
save_root = "augmented_dataset"
os.makedirs(save_root, exist_ok=True)

# Helper to apply paired augmentations
def generate_augmented_pairs(image, mask):
    aug_pairs = []

    # Resize originals
    image = TF.resize(image, resize_shape)
    mask = TF.resize(mask, resize_shape)

    # 1. Horizontal flip
    aug_pairs.append((TF.hflip(image), TF.hflip(mask)))

    # 2. Vertical flip
    aug_pairs.append((TF.vflip(image), TF.vflip(mask)))

    # 3. Random rotation (±15°) - same angle
    angle = random.uniform(-15, 15)
    aug_pairs.append((
        TF.rotate(image, angle),
        TF.rotate(mask, angle)
    ))

    # 4. Affine transformation - same parameters
    affine_angle = 10
    translate = (5, 5)
    scale = 1.05
    shear = [5.0, 0.0]  # X and Y shear
    aug_pairs.append((
        TF.affine(image, angle=affine_angle, translate=translate, scale=scale, shear=shear),
        TF.affine(mask, angle=affine_angle, translate=translate, scale=scale, shear=shear)
    ))

    # 5. Perspective transform - same start and end points
    startpoints = [(0, 0), (224, 0), (224, 224), (0, 224)]
    endpoints = [
        (random.randint(0, 10), random.randint(0, 10)),
        (224 - random.randint(0, 10), random.randint(0, 10)),
        (224 - random.randint(0, 10), 224 - random.randint(0, 10)),
        (random.randint(0, 10), 224 - random.randint(0, 10))
    ]
    aug_pairs.append((
        TF.perspective(image, startpoints, endpoints),
        TF.perspective(mask, startpoints, endpoints)
    ))

    # 6. Cropping and resizing - same crop box
    top, left, height, width = 20, 20, 180, 180
    aug_pairs.append((
        TF.resized_crop(image, top=top, left=left, height=height, width=width, size=resize_shape),
        TF.resized_crop(mask, top=top, left=left, height=height, width=width, size=resize_shape)
    ))

    # 7. Color jitter (image only)
    jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)
    aug_pairs.append((jitter(image), mask))

    # 8. Gaussian blur (image only)
    blur = TF.gaussian_blur(image, kernel_size=5)
    aug_pairs.append((blur, mask))

    return aug_pairs

# Loop through dataset
for class_name in ["benign", "malignant", "normal"]:
    input_dir = os.path.join(input_root, class_name)
    output_dir = os.path.join(save_root, class_name)
    os.makedirs(output_dir, exist_ok=True)

    original_count = 0
    total_augmented_count = 0

    for file in os.listdir(input_dir):
        if "_mask" in file or not file.endswith((".png", ".jpg", ".jpeg")):
            continue

        base_name = os.path.splitext(file)[0]
        image_path = os.path.join(input_dir, file)
        mask_path = os.path.join(input_dir, f"{base_name}_mask.png")

        if not os.path.exists(mask_path):
            print(f"❌ Missing mask for {file}, skipping.")
            continue

        original_count += 1

        # Load images
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Resize and save original
        image_tensor = TF.to_tensor(TF.resize(image, resize_shape))
        mask_tensor = TF.to_tensor(TF.resize(mask, resize_shape))
        save_image(image_tensor, os.path.join(output_dir, f"{base_name}.png"))
        save_image(mask_tensor, os.path.join(output_dir, f"{base_name}_mask.png"))

        # Generate augmented pairs
        aug_pairs = generate_augmented_pairs(image, mask)

        for i, (aug_img, aug_mask) in enumerate(aug_pairs):
            save_image(TF.to_tensor(aug_img), os.path.join(output_dir, f"{base_name}_aug{i+1}.png"))
            save_image(TF.to_tensor(aug_mask), os.path.join(output_dir, f"{base_name}_aug{i+1}_mask.png"))

        total_augmented_count += len(aug_pairs)

    total_final_images = original_count + total_augmented_count
    print(f"✅ Augmented {original_count} images of {class_name} class to {total_final_images} images")

print("🎉 All images and masks processed with advanced augmentation!")
