#!/usr/bin/env python3
import os, random, time, json
from collections import defaultdict, Counter
from dataclasses import dataclass

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm.auto import tqdm

# -------------------- 0) DEVICE PICK --------------------
def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    # Apple Silicon (PyTorch MPS)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), "mps"
    return torch.device("cpu"), "cpu"

DEVICE, DEVICE_STR = pick_device()

# -------------------- 1) CONFIG --------------------
@dataclass
class CFG:
    DATASET_DIR: str = r"data"      # change to your dataset root (contains subfolders 0..5)
    NUM_CLASSES: int = 6
    IMG_SIZE: int = 224
    BATCH_SIZE: int = 32            # safer default for laptops; raise if you have VRAM
    # Windows often prefers 0 workers; Linux/macOS can use cpu_count()-1
    NUM_WORKERS: int = 0 if os.name == "nt" else max(os.cpu_count() - 1, 0)
    EPOCHS: int = 20
    LR: float = 3e-4
    WEIGHT_DECAY: float = 1e-4
    LABEL_SMOOTHING: float = 0.1
    TRAIN_SPLIT: float = 0.8
    RANDOM_SEED: int = 42
    SAVE_DIR: str = "checkpoints"
    RUN_NAME: str = time.strftime("run_%Y%m%d_%H%M%S")
    EARLY_STOP_PATIENCE: int = 6
    FAST_DEV_RUN: bool = False      # True = run only a few batches to sanity-check
    MAX_TRAIN_BATCHES: int = 8
    MAX_VAL_BATCHES: int = 4

CFG = CFG()

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Repro
random.seed(CFG.RANDOM_SEED)
np.random.seed(CFG.RANDOM_SEED)
torch.manual_seed(CFG.RANDOM_SEED)
if DEVICE_STR == "cuda":
    torch.cuda.manual_seed_all(CFG.RANDOM_SEED)
    torch.backends.cudnn.benchmark = True

os.makedirs(CFG.SAVE_DIR, exist_ok=True)
print(f"Device: {DEVICE_STR} | CUDA: {torch.cuda.is_available()} | MPS: {getattr(torch.backends,'mps',None) and torch.backends.mps.is_available()}")

# -------------------- 2) DATA --------------------
def list_images_by_class(root):
    """Return list of (path, label_int). Label = folder name (0..5)."""
    samples = []
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Dataset root not found: {root}")
    for lbl in sorted(os.listdir(root)):
        lbl_path = os.path.join(root, lbl)
        if not os.path.isdir(lbl_path): 
            continue
        if not lbl.isdigit():
            continue
        y = int(lbl)
        for fn in os.listdir(lbl_path):
            if fn.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
                samples.append((os.path.join(lbl_path, fn), y))
    return samples

def group_key_from_name(path):
    """Filename prefix before first '_' to keep augmented variants together."""
    name = os.path.splitext(os.path.basename(path))[0]
    return name.split("_", 1)[0]

class ImageListDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, y = self.samples[idx]
        img = Image.open(p).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, y

def build_transforms():
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(CFG.IMG_SIZE, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.RandomRotation(25),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.RandomGrayscale(p=0.08),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    test_tf = transforms.Compose([
        transforms.Resize((CFG.IMG_SIZE, CFG.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, test_tf

def grouped_split(samples, train_ratio=0.8, seed=42):
    """Group-aware train/val split preventing near-duplicate leakage."""
    groups_by_label = defaultdict(lambda: defaultdict(list))  # label -> group_key -> idx list
    for i, (p, y) in enumerate(samples):
        g = group_key_from_name(p)
        groups_by_label[y][g].append(i)

    train_idx, val_idx = [], []
    rng = random.Random(seed)
    for _, group_map in groups_by_label.items():
        ks = list(group_map.keys())
        rng.shuffle(ks)
        cutoff = int(len(ks) * train_ratio)
        train_groups = set(ks[:cutoff])
        for g, idxs in group_map.items():
            (train_idx if g in train_groups else val_idx).extend(idxs)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx

# -------------------- 3) MODEL --------------------
def build_model(num_classes=6, pretrained=True):
    # Handle older/newer torchvision APIs
    try:
        from torchvision.models import ResNet18_Weights
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.resnet18(weights=weights)
    except Exception:
        m = models.resnet18(pretrained=pretrained)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

# -------------------- 4) TRAINING UTILS --------------------
def compute_class_weights(samples):
    counts = Counter([y for _, y in samples])
    max_c = max(counts.values())
    w = torch.tensor([max_c / counts.get(c, 1) for c in range(CFG.NUM_CLASSES)], dtype=torch.float32)
    return w

def accuracy(logits, targets):
    preds = logits.argmax(1)
    return (preds == targets).float().mean().item()

def make_loader(dataset, batch_size, shuffle, num_workers):
    pin = (DEVICE_STR == "cuda")
    kwargs = dict(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin)
    # persistent_workers/prefetch_factor only valid when num_workers > 0
    if num_workers and num_workers > 0:
        kwargs.update(dict(persistent_workers=True, prefetch_factor=2))
    return DataLoader(dataset, **kwargs)

# -------------------- 5) MAIN --------------------
def main():
    # data
    all_samples = list_images_by_class(CFG.DATASET_DIR)
    assert len(all_samples) > 0, f"No images found under {CFG.DATASET_DIR}"

    train_idx, val_idx = grouped_split(all_samples, CFG.TRAIN_SPLIT, CFG.RANDOM_SEED)
    train_samples = [all_samples[i] for i in train_idx]
    val_samples   = [all_samples[i] for i in val_idx]

    train_tf, test_tf = build_transforms()
    ds_train = ImageListDataset(train_samples, transform=train_tf)
    ds_val   = ImageListDataset(val_samples,   transform=test_tf)

    dl_train = make_loader(ds_train, CFG.BATCH_SIZE, True,  CFG.NUM_WORKERS)
    dl_val   = make_loader(ds_val,   CFG.BATCH_SIZE, False, CFG.NUM_WORKERS)

    # model / loss / opt
    model = build_model(CFG.NUM_CLASSES, pretrained=True).to(DEVICE)
    class_weights = compute_class_weights(train_samples).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=CFG.LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # AMP only on CUDA (MPS AMP is experimental; keep it simple)
    use_amp = (DEVICE_STR == "cuda")
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    best_val = float("inf")
    best_path = os.path.join(CFG.SAVE_DIR, f"{CFG.RUN_NAME}_best.pth")
    history, patience = [], CFG.EARLY_STOP_PATIENCE
    print(f"Train: {len(train_samples)}   Val: {len(val_samples)}   Saving to: {best_path}")

    for epoch in range(1, CFG.EPOCHS + 1):
        # ---------------- TRAIN ----------------
        model.train()
        train_loss, train_acc, n_train = 0.0, 0.0, 0

        for b_idx, (imgs, ys) in enumerate(tqdm(dl_train, desc=f"Epoch {epoch} [train]", leave=False)):
            imgs, ys = imgs.to(DEVICE, non_blocking=use_amp), ys.to(DEVICE, non_blocking=use_amp)
            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.amp.autocast('cuda', enabled=True):
                    logits = model(imgs)
                    loss = criterion(logits, ys)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(imgs)
                loss = criterion(logits, ys)
                loss.backward()
                optimizer.step()

            bs = ys.size(0)
            train_loss += loss.item() * bs
            train_acc  += accuracy(logits, ys) * bs
            n_train    += bs

            if CFG.FAST_DEV_RUN and (b_idx + 1) >= CFG.MAX_TRAIN_BATCHES:
                break

        train_loss /= max(1, n_train)
        train_acc  /= max(1, n_train)

        # ---------------- VAL ----------------
        model.eval()
        val_loss, val_acc, n_val = 0.0, 0.0, 0
        with torch.no_grad():
            for b_idx, (imgs, ys) in enumerate(tqdm(dl_val, desc=f"Epoch {epoch} [val]", leave=False)):
                imgs, ys = imgs.to(DEVICE, non_blocking=use_amp), ys.to(DEVICE, non_blocking=use_amp)
                if use_amp:
                    with torch.amp.autocast('cuda', enabled=True):
                        logits = model(imgs)
                        loss = criterion(logits, ys)
                else:
                    logits = model(imgs)
                    loss = criterion(logits, ys)

                bs = ys.size(0)
                val_loss += loss.item() * bs
                val_acc  += accuracy(logits, ys) * bs
                n_val    += bs

                if CFG.FAST_DEV_RUN and (b_idx + 1) >= CFG.MAX_VAL_BATCHES:
                    break

        val_loss /= max(1, n_val)
        val_acc  /= max(1, n_val)
        scheduler.step(val_loss)

        # epoch log
        info = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(info)
        print(json.dumps(info))

        # checkpoint
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            torch.save({
                "cfg": CFG.__dict__,
                "state_dict": model.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "class_weights": class_weights.detach().cpu().tolist(),
                "device": DEVICE_STR,
            }, best_path)
            patience = CFG.EARLY_STOP_PATIENCE
        else:
            patience -= 1
            if patience <= 0:
                print("Early stopping.")
                break

    # also save a final snapshot
    final_path = os.path.join(CFG.SAVE_DIR, f"{CFG.RUN_NAME}_final.pth")
    torch.save(model.state_dict(), final_path)
    with open(os.path.join(CFG.SAVE_DIR, f"{CFG.RUN_NAME}_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"Best: {best_val:.4f}  | saved: {best_path}")
    print(f"Final weights saved to: {final_path}")

if __name__ == "__main__":
    main()
