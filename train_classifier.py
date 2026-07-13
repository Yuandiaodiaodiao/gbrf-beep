import os
import shutil
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import SqueezeNet1_1_Weights, MobileNet_V3_Small_Weights
import time
from datetime import datetime

# Configuration
POSITIVE_DIR = "positive_candidates"
FRAMES_DIR = "frames"
NEW_DATA_DIR = "new_data"
NEGATIVE_DIR = "negative_samples"
MODEL_NAME = "squeezenet"  # Options: squeezenet, mobilenet
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-4
IMG_SIZE = 224
SEED = 42
NUM_WORKERS = 2  # Set to 0 if DataLoader multiprocessing fails on Windows
USE_AMP = True  # Mixed precision for faster training on RTX GPUs

np.random.seed(SEED)
torch.manual_seed(SEED)

def get_unique_positives():
    """Remove duplicate positive samples (with _conf suffix)"""
    files = [f for f in os.listdir(POSITIVE_DIR) if f.endswith('.jpg')]
    base_files = []
    seen = set()
    for f in files:
        # Extract base frame id like 'frame_053320_075.jpg'
        base = f.split('_conf')[0]
        if not base.endswith('.jpg'):
            base += '.jpg'
        if base not in seen:
            seen.add(base)
            base_files.append(base)
    return base_files

def prepare_negative_samples(positive_files, num_negatives=None):
    """Select negative samples from frames/ and new_data/ that are not in positives"""
    positive_bases = set(positive_files)
    
    # Collect negative candidates from both directories
    neg_candidates = []
    for src_dir in [FRAMES_DIR, NEW_DATA_DIR]:
        if os.path.exists(src_dir):
            for f in os.listdir(src_dir):
                if f.endswith('.jpg') and f not in positive_bases:
                    neg_candidates.append((f, src_dir))
    
    if num_negatives is None:
        num_negatives = len(positive_bases) * 3  # 3 negatives per positive
    
    # Sample negatives to have good variety
    if len(neg_candidates) > num_negatives:
        indices = np.random.choice(len(neg_candidates), num_negatives, replace=False).tolist()
        neg_candidates = [neg_candidates[i] for i in indices]
    
    return neg_candidates

def copy_negatives(neg_candidates, dst_dir):
    """Copy negative samples from their source directories to dst_dir"""
    os.makedirs(dst_dir, exist_ok=True)
    for f, src_dir in neg_candidates:
        shutil.copy(os.path.join(src_dir, f), os.path.join(dst_dir, f))

def shift_image(img, shift_pct):
    """Apply a random translation of up to shift_pct of image size (both x and y)."""
    w, h = img.size
    dx = np.random.uniform(-shift_pct, shift_pct) * w
    dy = np.random.uniform(-shift_pct, shift_pct) * h
    # Affine matrix: (a, b, c, d, e, f) for [x', y'] = [a x + b y + c, d x + e y + f]
    return img.transform(img.size, Image.AFFINE, (1, 0, dx, 0, 1, dy), resample=Image.BILINEAR)

def create_augmented_train_data(src_dir, dst_dir, shifts=[0.01, 0.02]):
    """Create original + shifted versions of each image in src_dir, save to dst_dir."""
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    os.makedirs(dst_dir, exist_ok=True)
    files = [f for f in os.listdir(src_dir) if f.endswith('.jpg')]
    for f in files:
        src_path = os.path.join(src_dir, f)
        img = Image.open(src_path).convert('RGB')
        base = f[:-4]
        # Save original
        shutil.copy(src_path, os.path.join(dst_dir, f))
        # Save shifted versions
        for i, shift in enumerate(shifts, start=1):
            shifted = shift_image(img, shift)
            shifted_path = os.path.join(dst_dir, f'{base}_shift{int(shift*100)}.jpg')
            shifted.save(shifted_path, 'JPEG', quality=95)
    print(f"  Augmented {len(files)} images -> {len(os.listdir(dst_dir))} files in {dst_dir}")

def copy_samples(files, src_dir, dst_dir):
    """Copy samples from a single source directory to dst_dir"""
    os.makedirs(dst_dir, exist_ok=True)
    for f in files:
        shutil.copy(os.path.join(src_dir, f), os.path.join(dst_dir, f))

class TextDataset(Dataset):
    def __init__(self, data_dir, label, transform=None):
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]
        self.data_dir = data_dir
        self.label = label
        self.transform = transform
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.data_dir, self.files[idx])).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.label

def get_model(model_name):
    if model_name == 'squeezenet':
        model = models.squeezenet1_1(weights=SqueezeNet1_1_Weights.IMAGENET1K_V1)
        # Replace classifier with our own
        model.classifier[1] = nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1))
        model.num_classes = 2
    elif model_name == 'mobilenet':
        model = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    return model

def train():
    # Data preparation
    print("Preparing dataset...")
    positive_files = get_unique_positives()
    print(f"Unique positive samples: {len(positive_files)}")
    
    neg_candidates = prepare_negative_samples(positive_files)
    print(f"Selected negative samples: {len(neg_candidates)}")
    
    # Split into train/val
    np.random.shuffle(positive_files)
    np.random.shuffle(neg_candidates)
    
    pos_split = int(len(positive_files) * 0.8)
    neg_split = int(len(neg_candidates) * 0.8)
    
    pos_train, pos_val = positive_files[:pos_split], positive_files[pos_split:]
    neg_train, neg_val = neg_candidates[:neg_split], neg_candidates[neg_split:]
    
    # Copy to temporary directories
    if os.path.exists('train_data'):
        shutil.rmtree('train_data')
    if os.path.exists('val_data'):
        shutil.rmtree('val_data')
    
    copy_samples(pos_train, POSITIVE_DIR, 'train_data/positive')
    copy_negatives(neg_train, 'train_data/negative')
    copy_samples(pos_val, POSITIVE_DIR, 'val_data/positive')
    copy_negatives(neg_val, 'val_data/negative')
    
    # Generate explicit augmented training data: original + +/-1% + +/-2% shifts
    print("Generating augmented training data...")
    create_augmented_train_data('train_data/positive', 'train_data_aug/positive', shifts=[0.01, 0.02])
    create_augmented_train_data('train_data/negative', 'train_data_aug/negative', shifts=[0.01, 0.02])
    
    # Transforms
    # No online augmentation; augmentation is already baked into train_data_aug.
    # No color change, no rotation, no flip.
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Datasets
    train_pos = TextDataset('train_data_aug/positive', label=1, transform=train_transform)
    train_neg = TextDataset('train_data_aug/negative', label=0, transform=train_transform)
    val_pos = TextDataset('val_data/positive', label=1, transform=val_transform)
    val_neg = TextDataset('val_data/negative', label=0, transform=val_transform)
    
    train_dataset = torch.utils.data.ConcatDataset([train_pos, train_neg])
    val_dataset = torch.utils.data.ConcatDataset([val_pos, val_neg])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    
    # Model
    print(f"Loading {MODEL_NAME} model...")
    model = get_model(MODEL_NAME)
    
    # Freeze early layers
    if MODEL_NAME == 'squeezenet':
        for param in model.features.parameters():
            param.requires_grad = False
    elif MODEL_NAME == 'mobilenet':
        for param in model.features.parameters():
            param.requires_grad = False
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    model = model.to(device)
    
    # Class weights: positives are minority and false negatives are costly, so weight them higher.
    pos_weight = len(train_neg) / len(train_pos)
    class_weights = torch.tensor([1.0, pos_weight], dtype=torch.float32).to(device)
    print(f"Using class weights: [neg={class_weights[0]:.2f}, pos={class_weights[1]:.2f}]")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    
    # AMP scaler for mixed precision training
    scaler = torch.amp.GradScaler('cuda') if (USE_AMP and device.type == 'cuda') else None
    if scaler is not None:
        print("Using Automatic Mixed Precision (AMP) for training")
    
    best_val_acc = 0.0
    best_model_path = f'jzsz_classifier_{MODEL_NAME}.pth'
    
    print("Training...")
    train_start = time.time()
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        print(f"Epoch {epoch+1}/{EPOCHS}: train_loss={train_loss/len(train_loader):.4f}, "
              f"train_acc={train_acc:.1f}%, val_loss={val_loss/len(val_loader):.4f}, val_acc={val_acc:.1f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> Saved best model with val_acc={val_acc:.1f}%")
    
    train_elapsed = time.time() - train_start
    print(f"Training complete in {train_elapsed:.1f}s. Best val accuracy: {best_val_acc:.1f}%")
    
    # Export to ONNX for fast inference
    print("Exporting to ONNX...")
    # Load model on CPU for ONNX export
    model = get_model(MODEL_NAME)
    model.load_state_dict(torch.load(best_model_path, map_location='cpu'))
    model = model.to('cpu')
    model.eval()
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    onnx_path = best_model_path.replace('.pth', '.onnx')
    torch.onnx.export(model, dummy_input, onnx_path,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
                      opset_version=11,
                      dynamo=False)
    print(f"Exported to {onnx_path}")
    
    return onnx_path

if __name__ == '__main__':
    train()
