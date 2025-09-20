import numpy as np
import math
from scipy.ndimage import label
from sklearn.metrics import f1_score,accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

# === Relaxed 10-second rule ===
def apply_10s_relaxed(gt_labels, pred_labels, fps=6, num_phases=7):
    oriT = 10 * fps
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

            if iPhase in [4, 5]:
                head = cur_diff[:t]
                tail = cur_diff[-t:]
                cur_diff[:t][head == -1] = 0
                cur_diff[-t:][(tail == 1) | (tail == 2)] = 0
            elif iPhase in [6, 7]:
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

# === Most frequent helper ===
def most_frequent(lst):
    return max(set(lst), key=lst.count)

# === Load file ===
with open(r"D:\Research\Phong\Cholec80_Project_4\GJ\cholec80_train_2ToolB_a_0_1_filter3_v1_last.txt", "r") as f:
    lines_raw = f.readlines()

lines = []
for line in lines_raw:
    frame_num = line.split("Frame")[1].split(",")[0]
    lines.append(
        f'{line.split("Frame")[0]}Frame{str(frame_num).zfill(6)}, {line.split(", ")[1]}, {line.split(", ")[2]}'
    )
lines.sort()

# === Parse GT and prediction ===
y_true = [int(line.split(",")[1].split("[")[1].split("]")[0]) for line in lines]
y_pred = [int(line.split(",")[2].split("[")[1].split("]")[0]) for line in lines]

# === Search for best window size ===
f1_scores = []
accuracy_scores = []

for window_size in range(2, 200):
    l = math.floor(window_size / 2)

    moving_average = []
    i = l
    while i < len(y_pred) - l:
        window = y_pred[i - l : i + l]
        window_mode = most_frequent(window)
        moving_average.append(round(window_mode))
        i += 1

    # Use the same structure as the original code
    smoothed_pred = y_pred[:l] + moving_average + y_pred[-l:]

    # Apply relaxed boundary
    y_pred_relaxed = apply_10s_relaxed(y_true, smoothed_pred)

    f1 = f1_score(y_true, y_pred_relaxed, average="macro")
    f1_scores.append(f1)
    print('window_size=', window_size, 'f1=', f1)

    accuracy = accuracy_score(y_true, y_pred_relaxed)
    accuracy_scores.append(accuracy)

# === Find best window size ===
best_window = np.argmax(f1_scores) + 2# start w=2
best_f1 = f1_scores[best_window - 1]

# best_window2 = np.argmax(accuracy_scores) + 2
# best_accuracy = accuracy_scores[best_window2 - 1]

# === Plot ===
# plt.figure(figsize=(10, 5))
# plt.plot(range(1, 201), f1_scores)
# plt.axvline(best_window, color='r', linestyle='--', label=f"Best: {best_window}")
# plt.xlabel("Window Size")
# plt.ylabel("Macro F1 Score (Relaxed)")
# plt.title("Window Size Search for Best F1 Score")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

print(f"✅ Best window size: {best_window}, Macro F1 Score: {best_f1:.4f}")
# print(f"✅ Best window size: {best_window2}, Accuracy Score: {best_accuracy:.4f}")


# print the metrics for the best F1 score
from sklearn.metrics import precision_score, recall_score

# === Recompute smoothed predictions using best F1 window ===
l = math.floor(best_window / 2)
moving_average = []
i = l
while i < len(y_pred) - l:
    window = y_pred[i - l : i + l]
    window_mode = most_frequent(window)
    moving_average.append(round(window_mode))
    i += 1

# Pad beginning and end with original predictions
smoothed_pred_best = y_pred[:l] + moving_average + y_pred[-l:]

# === Metrics BEFORE relaxed 10s rule ===
acc_before = accuracy_score(y_true, smoothed_pred_best)
prec_before = precision_score(y_true, smoothed_pred_best, average="macro")
rec_before = recall_score(y_true, smoothed_pred_best, average="macro")
f1_before = f1_score(y_true, smoothed_pred_best, average="macro")

# === Apply relaxed 10s rule ===
y_pred_relaxed_best = apply_10s_relaxed(y_true, smoothed_pred_best)

# === Metrics AFTER relaxed 10s rule ===
acc_after = accuracy_score(y_true, y_pred_relaxed_best)
prec_after = precision_score(y_true, y_pred_relaxed_best, average="macro")
rec_after = recall_score(y_true, y_pred_relaxed_best, average="macro")
f1_after = f1_score(y_true, y_pred_relaxed_best, average="macro")

# === Print both ===
print(f"\n📊 Metrics BEFORE applying 10s relaxed rule (Window = {best_window}):")
print(f"   Accuracy : {acc_before:.4f}")
print(f"   Precision: {prec_before:.4f}")
print(f"   Recall   : {rec_before:.4f}")
print(f"   F1 Score : {f1_before:.4f}")

print(f"\n📊 Metrics AFTER applying 10s relaxed rule (Window = {best_window}):")
print(f"   Accuracy : {acc_after:.4f}")
print(f"   Precision: {prec_after:.4f}")
print(f"   Recall   : {rec_after:.4f}")
print(f"   F1 Score : {f1_after:.4f}")


# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred_relaxed_best)
cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)

num_classes = cm.shape[0]
class_labels = [f'P{i}' for i in range(num_classes)]  # P0 to P6

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)

# Title and labels
ax.set_xlabel('Predicted Labels', fontsize=12)
ax.set_ylabel('True Labels', fontsize=12)
ax.set_xticks(np.arange(num_classes))
ax.set_yticks(np.arange(num_classes))
ax.set_xticklabels(class_labels)
ax.set_yticklabels(class_labels)
plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
plt.setp(ax.get_yticklabels(), rotation=0, va="center")

# Show cell values
for i in range(num_classes):
    for j in range(num_classes):
        value = cm_normalized[i, j]
        ax.text(j, i, f'{value:.2f}', ha="center", va="center",
                color="black" if value < 0.5 else "white", fontsize=10)

# Colorbar
fig.colorbar(im, ax=ax)
plt.tight_layout()
plt.show()
