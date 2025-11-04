import os
import re
import cv2
import random
import numpy as np

# ========== CONFIG ==========
INPUT_DIR = "."
OUTPUT_DIR = "Project/new_dataset"
AUG_PER_IMAGE = 5
CREATE_LABEL_SUBFOLDERS = True
# =============================

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_label_from_filename(filename: str):
    """
    Extracts the number of fingers (0–5) from the filename or folder name.
    """
    match = re.search(r"([0-5])", filename)
    return match.group(1) if match else None

def random_rotation(img):
    """Apply a random rotation between -30° and 30°."""
    angle = random.uniform(-30, 30)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h)), f"rot{int(angle)}"

def random_brightness(img):
    """Adjust brightness randomly."""
    value = random.randint(-50, 50)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + value, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), f"bright{value}"

def random_blur(img):
    """Apply random Gaussian blur."""
    k = random.choice([3, 5])
    return cv2.GaussianBlur(img, (k, k), 0), f"blur{k}"

def random_noise(img):
    """Add Gaussian noise to the image."""
    mean = 0
    std = random.randint(5, 25)
    noise = np.random.normal(mean, std, img.shape).astype(np.float32)
    noisy_img = cv2.add(img.astype(np.float32), noise)
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img, f"noise{std}"

def augment_image(img):
    """Apply a random combination of transformations."""
    transformations = []
    img_out = img.copy()

    # Flip
    if random.random() > 0.5:
        img_out = cv2.flip(img_out, 1)
        transformations.append("flip")

    # Rotation
    if random.random() > 0.3:
        img_out, tag = random_rotation(img_out)
        transformations.append(tag)

    # Brightness
    if random.random() > 0.4:
        img_out, tag = random_brightness(img_out)
        transformations.append(tag)

    # Blur
    if random.random() > 0.4:
        img_out, tag = random_blur(img_out)
        transformations.append(tag)

    # Noise
    if random.random() > 0.4:
        img_out, tag = random_noise(img_out)
        transformations.append(tag)

    return img_out, "_".join(transformations) if transformations else "orig"

# ========== MAIN LOOP ==========
for root, _, files in os.walk(INPUT_DIR):
    for filename in files:
        if filename.lower().endswith(".png"):
            label = get_label_from_filename(filename)
            if label is None:
                print(f"⚠️ Skipping {filename} (no label found)")
                continue

            path = os.path.join(root, filename)
            img = cv2.imread(path)
            if img is None:
                print(f"⚠️ Could not read {filename}")
                continue

            # Create label folder if needed
            save_dir = os.path.join(OUTPUT_DIR, label) if CREATE_LABEL_SUBFOLDERS else OUTPUT_DIR
            os.makedirs(save_dir, exist_ok=True)

            base = os.path.splitext(filename)[0]

            # Save original
            cv2.imwrite(os.path.join(save_dir, f"{label}_orig.png"), img)

            # Save augmented versions
            for i in range(AUG_PER_IMAGE):
                aug_img, tag = augment_image(img)
                out_name = f"{label}_{tag}_{i}.png"
                cv2.imwrite(os.path.join(save_dir, out_name), aug_img)

print(f"\n✅ Dataset built successfully in '{OUTPUT_DIR}/'")
