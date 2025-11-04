import os
import random
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# --- 1) PARAMETERS ---

DATASET_DIR = r'Project/new_dataset'   # folder that has subfolders 0,1,2,3,4,5
IMG_SIZE = 128
BATCH_SIZE = 32
NUM_CLASSES = 6   # 0..5
EPOCHS = 10
LEARNING_RATE = 1e-3
NUM_WORKERS = 4
TRAIN_SPLIT = 0.8
RANDOM_SEED = 42


# --- 2) INDEX THE DATA ---

def index_dataset(root_dir):
    """
    Walk new_dataset/, expecting:
       new_dataset/
          0/*.png
          1/*.png
          ...
          5/*.png
    Returns [(absolute_img_path, label_int), ...]
    """
    samples = []

    root_dir = os.path.abspath(root_dir)

    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Directory does not exist: {root_dir}")

    for label_name in sorted(os.listdir(root_dir)):
        class_dir = os.path.join(root_dir, label_name)
        if not os.path.isdir(class_dir):
            continue

        # interpret folder name as label
        try:
            label_int = int(label_name)
        except ValueError:
            continue  # skip weird folders

        for fname in os.listdir(class_dir):
            if fname.lower().endswith(".png"):
                img_path = os.path.join(class_dir, fname)
                img_path = os.path.abspath(img_path)
                img_path = os.path.normpath(img_path)
                samples.append((img_path, label_int))

    if len(samples) == 0:
        raise RuntimeError(
            f"No images found in {root_dir}. "
            f"Expected structure like {root_dir}/0/img.png"
        )

    return samples


def split_dataset(samples, train_ratio=TRAIN_SPLIT, seed=RANDOM_SEED):
    random.Random(seed).shuffle(samples)
    split_idx = int(len(samples) * train_ratio)
    train_samples = samples[:split_idx]
    test_samples = samples[split_idx:]
    return train_samples, test_samples


# --- 3) DATASET CLASS ---

class FingerDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        # img_path is now absolute, so no more "Project/Project" issue
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


# --- 4) MODEL ---

class FingerCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # after 3 pools: 128 -> 64 -> 32 -> 16
        self.fc1 = nn.Linear(128 * 16 * 16, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 128->64
        x = self.pool(F.relu(self.conv2(x)))  # 64->32
        x = self.pool(F.relu(self.conv3(x)))  # 32->16
        x = x.view(-1, 128 * 16 * 16)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# --- 5) TRANSFORMS / DATALOADERS ---

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])
    return train_transform, test_transform


def get_dataloaders(root_dir, num_workers=NUM_WORKERS):
    all_samples = index_dataset(root_dir)
    train_samples, test_samples = split_dataset(all_samples)

    train_tf, test_tf = get_transforms()
    train_ds = FingerDataset(train_samples, transform=train_tf)
    test_ds  = FingerDataset(test_samples,  transform=test_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, test_loader, train_ds, test_ds


# --- 6) EVAL LOOP ---

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    acc = 100.0 * correct / total if total else 0.0
    return acc


# --- 7) MAIN TRAIN LOOP ---

def main():
    train_loader, test_loader, train_ds, test_ds = get_dataloaders(DATASET_DIR)

    print(f"Found {len(train_ds)} training images and {len(test_ds)} test images.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = FingerCNN().to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 100 == 0:
                avg_loss = running_loss / 100.0
                print(f"[Epoch {epoch+1}, Batch {i+1}] loss: {avg_loss:.3f}")
                running_loss = 0.0

        val_acc = evaluate(model, test_loader, device)
        print(f"*** Epoch {epoch+1} Validation Accuracy: {val_acc:.2f}% ***")

    print("Finished Training!")
    torch.save(model.state_dict(), "finger_counter_torch.pth")
    print("Model saved to 'finger_counter_torch.pth'")


if __name__ == "__main__":
    main()
