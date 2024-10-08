import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms as T
import segmentation_models_pytorch as smp
from utils import MicroscopyDataset
import numpy as np
import matplotlib.pyplot as plt

# hyperparameters
BATCH_SIZE = 4
NUM_CLASSES = 3  #(e.g., background, cilia, nuclei)
EPOCHS = 50
LEARNING_RATE = 1e-4
TRAIN_SPLIT = 0.8
VALID_SPLIT = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "models/unet_model.pth"

# Transforms
transform = T.Compose([
    T.Resize((256, 256)),  # resize images to 256x256 (need to figure out how to enhance size or split to patches)
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # normalization
])

# dataset
dataset = MicroscopyDataset(images_dir='data/images/', masks_dir='data/masks/', transform=transform)

# split dataset
train_size = int(TRAIN_SPLIT * len(dataset))
valid_size = len(dataset) - train_size
train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

# data Loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# model itself
# we can choose a pre-trained encoder, 'resnet34' for example
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",  # pre-trained weights
    in_channels=1,  # grayscale
    classes=NUM_CLASSES
)

model = model.to(DEVICE)

# loss and optimizer
# gonna use CrossEntropyLoss for multiclass segmentation
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# metrics
def calculate_iou(pred, target, num_classes):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).long().sum().item()
        union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
        if union == 0:
            ious.append(float('nan'))  # If no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / float(max(union, 1)))
    return ious


# training Loop
best_val_loss = float('inf')

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for images, masks in train_loader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

    train_loss = train_loss / len(train_loader.dataset)

    # validation
    model.eval()
    val_loss = 0.0
    iou_scores = []
    with torch.no_grad():
        for images, masks in valid_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)
            loss = loss_fn(outputs, masks)
            val_loss += loss.item() * images.size(0)

            preds = torch.argmax(outputs, dim=1)
            ious = calculate_iou(preds, masks, NUM_CLASSES)
            iou_scores.append(ious)

    val_loss = val_loss / len(valid_loader.dataset)
    iou_scores = np.nanmean(iou_scores, axis=0)
    mean_iou = np.nanmean(iou_scores)

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f} Mean IoU: {mean_iou:.4f}")

    # saving the best obtained model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print("Model saved!")

print("Training Completed.")
