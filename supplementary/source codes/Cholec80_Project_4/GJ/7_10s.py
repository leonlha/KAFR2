import numpy as np
from scipy.ndimage import label
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def apply_10s_relaxed(gt_labels, pred_labels, fps=6, num_phases=7):
    oriT = 10 * fps  # 10 seconds in frames
    diff = [p - g for p, g in zip(pred_labels, gt_labels)]
    updated_diff = diff.copy()

    gt_labels_np = np.array(gt_labels)
    diff_np = np.array(diff)

    for iPhase in range(1, num_phases + 1):
        mask = (gt_labels_np == iPhase)
        labeled_array, num_features = label(mask)

        for iConn in range(1, num_features + 1):
            indices = np.where(labeled_array == iConn)[0]
            if len(indices) == 0:
                continue

            start_idx = indices[0]
            end_idx = indices[-1]
            cur_diff = diff_np[start_idx:end_idx + 1].copy()
            t = min(oriT, len(cur_diff))

            if iPhase in [4, 5]:  # Gallbladder dissection/packaging
                head = cur_diff[:t]
                tail = cur_diff[-t:]
                cur_diff[:t][head == -1] = 0
                cur_diff[-t:][(tail == 1) | (tail == 2)] = 0
            elif iPhase in [6, 7]:  # CleaningCoagulation, Retraction
                head = cur_diff[:t]
                tail = cur_diff[-t:]
                cur_diff[:t][(head == -1) | (head == -2)] = 0
                cur_diff[-t:][(tail == 1) | (tail == 2)] = 0
            else:
                head = cur_diff[:t]
                tail = cur_diff[-t:]
                cur_diff[:t][head == -1] = 0
                cur_diff[-t:][tail == 1] = 0

            updated_diff[start_idx:end_idx + 1] = cur_diff

    relaxed_pred = []
    for i in range(len(pred_labels)):
        if updated_diff[i] == 0:
            relaxed_pred.append(gt_labels[i])
        else:
            relaxed_pred.append(pred_labels[i])

    return relaxed_pred


# === MAIN EXECUTION ===

input_file = r"D:\Research\Phong\Cholec80_Project_4\GJ\cholec80_train_2Tool_0_1_filter3_v2_127.txt"

# Load predictions
with open(input_file, "r") as f:
    raw_lines = f.readlines()

lines = []
for line in raw_lines:
    frame_num = line.split("Frame")[1].split(",")[0]
    lines.append(
        f'{line.split("Frame")[0]}Frame{str(frame_num).zfill(6)}, {line.split(", ")[1]}, {line.split(", ")[2]}'
    )
lines.sort()

# Parse GT and predictions
y_true, y_pred = [], []
for line in lines:
    y_true.append(int(line.split(",")[1].strip()[1:-1]))
    y_pred.append(int(line.split(",")[2].strip()[1:-1]))

# Apply relaxed boundary
y_pred_relaxed = apply_10s_relaxed(y_true, y_pred, fps=6, num_phases=7)

# Evaluation
acc = accuracy_score(y_true, y_pred_relaxed)
f1 = f1_score(y_true, y_pred_relaxed, average="macro")
jaccard = jaccard_score(y_true, y_pred_relaxed, average=None)

print(f"Accuracy: {acc:.4f}")
print(f"Macro F1-score: {f1:.4f}")
print(f"Jaccard (per class): {np.round(jaccard, 3)}")
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred_relaxed))

# Plot confusion matrix
cf_matrix = confusion_matrix(y_true, y_pred_relaxed, normalize="true")
plt.figure(figsize=(8, 6))
sns.heatmap(cf_matrix, annot=True, cmap="Blues", fmt=".2f")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (10s Relaxed)")
plt.tight_layout()
# plt.show()
