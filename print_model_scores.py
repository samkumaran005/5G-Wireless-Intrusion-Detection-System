import torch
import numpy as np

# Load saved model checkpoint
checkpoint = torch.load('htstcl_gnn_models/htstcl_gnn_best.pth', map_location='cpu', weights_only=False)
history = checkpoint['history']

# Best model info stored in checkpoint
best_epoch = checkpoint.get('epoch', None)
best_val_acc = checkpoint.get('val_accuracy', None)
best_val_precision = checkpoint.get('val_precision', checkpoint.get('precision', None))
best_val_f1 = checkpoint.get('val_f1', checkpoint.get('f1_score', checkpoint.get('val_f1score', None)))
best_val_recall = checkpoint.get('val_recall', checkpoint.get('recall', None))

print("=" * 58)
print("=== BEST MODEL (Saved Checkpoint) ===")
print("=" * 58)
print(f"Best Epoch               : {best_epoch}")
if best_val_acc is not None:
    print(f"Best Validation Accuracy : {best_val_acc:.2f}%")
else:
    print("Best Validation Accuracy : N/A")

if best_val_precision is not None:
    print(f"Best Validation Precision: {best_val_precision:.4f}")
else:
    print("Best Validation Precision: N/A")

if best_val_f1 is not None:
    print(f"Best Validation F1 Score : {best_val_f1:.4f}")
else:
    print("Best Validation F1 Score : N/A")

if best_val_recall is not None:
    print(f"Best Validation Recall   : {best_val_recall:.4f}")
else:
    print("Best Validation Recall   : N/A")

print()
print("=" * 58)
print("=== AVERAGE SCORES ACROSS ALL EPOCHS ===")
print("=" * 58)
print(f"Total Epochs             : {len(history['val_accuracy'])}")
print(f"Average Training Acc     : {np.mean(history['train_accuracy']):.2f}%")
print(f"Average Validation Acc   : {np.mean(history['val_accuracy']):.2f}%")
print(f"Average Training Loss    : {np.mean(history['train_loss']):.4f}")
print(f"Average Validation Loss  : {np.mean(history['val_loss']):.4f}")

val_prec_list = history.get('val_precision', history.get('precision', []))
val_f1_list = history.get('val_f1', history.get('f1_score', history.get('val_f1score', [])))
val_recall_list = history.get('val_recall', history.get('recall', []))

if len(val_prec_list) > 0:
    print(f"Average Val Precision    : {np.mean(val_prec_list):.4f}")
else:
    print("Average Val Precision    : N/A")

if len(val_f1_list) > 0:
    print(f"Average Val F1 Score     : {np.mean(val_f1_list):.4f}")
else:
    print("Average Val F1 Score     : N/A")

if len(val_recall_list) > 0:
    print(f"Average Val Recall       : {np.mean(val_recall_list):.4f}")
else:
    print("Average Val Recall       : N/A")

print()
print("=" * 105)
print("=== EPOCH-BY-EPOCH ACCURACY ===")
print("=" * 105)
header = f"{'Epoch':<8} {'Train Acc':>12} {'Val Acc':>12} {'Train Loss':>12} {'Val Loss':>10} {'Precision':>11} {'F1 Score':>10} {'Recall':>9}"
print(header)
print("-" * 105)

for i in range(len(history['val_accuracy'])):
    marker = " <<< BEST" if (best_epoch is not None and i + 1 == best_epoch) else ""
    prec_val = f"{val_prec_list[i]:>11.4f}" if i < len(val_prec_list) else f"{'N/A':>11}"
    f1_val = f"{val_f1_list[i]:>10.4f}" if i < len(val_f1_list) else f"{'N/A':>10}"
    rec_val = f"{val_recall_list[i]:>9.4f}" if i < len(val_recall_list) else f"{'N/A':>9}"
    
    print(
        f"{i+1:<8} "
        f"{history['train_accuracy'][i]:>11.2f}% "
        f"{history['val_accuracy'][i]:>11.2f}% "
        f"{history['train_loss'][i]:>12.4f} "
        f"{history['val_loss'][i]:>10.4f} "
        f"{prec_val} "
        f"{f1_val} "
        f"{rec_val}"
        f"{marker}"
    )

print("-" * 105)
avg_prec_str = f"{np.mean(val_prec_list):>11.4f}" if len(val_prec_list) > 0 else f"{'N/A':>11}"
avg_f1_str = f"{np.mean(val_f1_list):>10.4f}" if len(val_f1_list) > 0 else f"{'N/A':>10}"
avg_rec_str = f"{np.mean(val_recall_list):>9.4f}" if len(val_recall_list) > 0 else f"{'N/A':>9}"

print(f"{'AVG':<8} "
      f"{np.mean(history['train_accuracy']):>11.2f}% "
      f"{np.mean(history['val_accuracy']):>11.2f}% "
      f"{np.mean(history['train_loss']):>12.4f} "
      f"{np.mean(history['val_loss']):>10.4f} "
      f"{avg_prec_str} "
      f"{avg_f1_str} "
      f"{avg_rec_str}")
