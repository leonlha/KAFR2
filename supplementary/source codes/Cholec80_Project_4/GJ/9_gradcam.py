
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from torch.utils.data import DataLoader
from models import CNN_3D
from dataloaders_PJ import PJVideoDataset
from data_augmentation import transform_val

# === SETUP ===
num_classes = 7
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_3D(num_classes).to(device)
model.load_state_dict(torch.load("./runs/cholec80_train_2Tool_0_4_filter3.pth", map_location=device))
model.eval()

# === HOOK SETUP ===
features = None
gradients = None

def forward_hook(module, input, output):
    global features
    features = output.detach()

def backward_hook(module, grad_in, grad_out):
    global gradients
    gradients = grad_out[0].detach()

# Attach hooks to last 3D conv block
target_layer = model.model.blocks[4]
target_layer.register_forward_hook(forward_hook)
target_layer.register_backward_hook(backward_hook)

# === LOAD A SAMPLE ===
test_dataset = PJVideoDataset(
    data_dir="D:/Research/Cholec80/images",
    transform=transform_val,
    positional_encoding=True,
    train=False,
    split="test",
    clip_length=16,
)
loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
inputs, labels, frame_nums, paths = next(iter(loader))

print(paths)
print(frame_nums)
print(labels)

inputs, frame_nums = inputs.to(device), frame_nums.to(device)



# === FORWARD + BACKWARD ===
outputs = model(inputs, frame_nums)[0]
pred_class = outputs.argmax(dim=1)
outputs[0, pred_class].backward()

# === GRAD-CAM CALCULATION ===
weights = gradients.mean(dim=[2, 3, 4], keepdim=True)
cam = F.relu((weights * features).sum(dim=1)).squeeze().cpu()
cam = [(c - c.min()) / (c.max() - c.min() + 1e-8) for c in cam]

# === ORIGINAL FRAMES ===
frames = inputs.squeeze().permute(1, 2, 3, 0).cpu().numpy()  # [T, H, W, C]
frame_scores = [c.sum().item() for c in cam]
top_indices = np.argsort(frame_scores)[::-1][:4]  # Top 4 frames

# === PLOT ===
# top_indices = sorted(top_indices[:4])
top_indices = sorted(top_indices[:4], reverse=True) # reverse indices

num_frames = len(top_indices)
fig, axs = plt.subplots(1, num_frames, figsize=(4 * num_frames, 5))

for i, t in enumerate(top_indices):
    frame = (frames[t] * 255).astype(np.uint8)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    heatmap = cv2.resize(cam[t].numpy(), (224, 224), interpolation=cv2.INTER_LINEAR)
    heatmap = (255 * heatmap).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

    axs[i].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axs[i].axis("off")
    axs[i].text(0.5, -0.1, f"[index={t}]", fontsize=21, ha='center', transform=axs[i].transAxes)

# Add the image path as second row, centered across figure
shared_path = os.path.basename(paths[0])
fig.text(0.5, 0.01, shared_path, ha='center', fontsize=21)

plt.subplots_adjust(bottom=0.1)  # make space for the second row
plt.show()