import os
import random
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from collections import defaultdict

# --- 1) PARAMETERS ---

DATASET_DIR = r'new_dataset'   # folder that has subfolders 0,1,2,3,4,5
IMG_SIZE = 128
BATCH_SIZE = 32
NUM_CLASSES = 6   # 0..5
EPOCHS = 3
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
    # group all variants of the same base image so they don't cross splits
    groups = defaultdict(list)
    for path, label in samples:
        stem = os.path.splitext(os.path.basename(path))[0]
        base = stem.split('_')[0]  # e.g., "0" from "0_flip_rot0_blur5_2"
        groups[(label, base)].append((path, label))

    rng = random.Random(seed)
    keys = list(groups.keys()); rng.shuffle(keys)
    split_idx = int(len(keys) * train_ratio)
    train_keys = set(keys[:split_idx])

    train_samples, test_samples = [], []
    for k, items in groups.items():
        (train_samples if k in train_keys else test_samples).extend(items)
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
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)      # <-- this was missing
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1   = nn.Linear(128 * 16 * 16, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2   = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 16 * 16)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)



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
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=0.5),
        transforms.RandomGrayscale(p=0.3),
        transforms.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
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

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)


    best_acc = 0.0
    best_path = "finger_counter_torch_best.pth"

    print("Starting training.")
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
        print(f"Epoch {epoch+1}: val_acc={val_acc:.2f}%  lr={scheduler.get_last_lr()[0]:.6f}")

        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)
            print(f"✓ Saved new best model to {best_path}")

    print(f"Finished Training! Best validation accuracy: {best_acc:.2f}%")
       
if __name__ == "__main__":
    main()
