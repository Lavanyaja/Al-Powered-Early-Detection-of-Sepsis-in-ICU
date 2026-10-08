"""
Step 7: Train LSTM Model on Full Dataset
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from lstm_model import build_lstm_model

SEQ_DIR = "processed_data/sequences"
MODEL_DIR = "models"
FIG_DIR = "figures"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

print("Loading sequence data...")
X_train = np.load(f"{SEQ_DIR}/X_train.npy")
y_train = np.load(f"{SEQ_DIR}/y_train.npy")
mask_train = np.load(f"{SEQ_DIR}/mask_train.npy")

X_val = np.load(f"{SEQ_DIR}/X_val.npy")
y_val = np.load(f"{SEQ_DIR}/y_val.npy")
mask_val = np.load(f"{SEQ_DIR}/mask_val.npy")

print(f"X_train: {X_train.shape}, X_val: {X_val.shape}")

real_y = y_train[mask_train == 1]
n_pos = real_y.sum()
n_neg = len(real_y) - n_pos
n_total = len(real_y)

weight_pos = n_total / (2.0 * n_pos)
weight_neg = n_total / (2.0 * n_neg)
print(f"\nReal training timesteps: {n_total:,}")
print(f"  Positive (sepsis): {int(n_pos):,} -> weight {weight_pos:.2f}")
print(f"  Negative (no sepsis): {int(n_neg):,} -> weight {weight_neg:.2f}")


def make_sample_weights(y, mask):
    class_w = np.where(y == 1, weight_pos, weight_neg)
    return (class_w * mask).astype(np.float32)


sw_train = make_sample_weights(y_train, mask_train)
sw_val = make_sample_weights(y_val, mask_val)

seq_len = X_train.shape[1]
n_features = X_train.shape[2]
model = build_lstm_model(seq_len, n_features)

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_auc", mode="max", patience=5, restore_best_weights=True
    ),
    keras.callbacks.ModelCheckpoint(
        f"{MODEL_DIR}/best_model.keras", monitor="val_auc", mode="max",
        save_best_only=True
    ),
    keras.callbacks.CSVLogger(f"{MODEL_DIR}/training_log.csv"),
]

print("\nStarting training (this will take a while on CPU - be patient)...")
history = model.fit(
    X_train, y_train,
    sample_weight=sw_train,
    validation_data=(X_val, y_val, sw_val),
    epochs=30,
    batch_size=64,
    callbacks=callbacks,
    verbose=1,
)

model.save(f"{MODEL_DIR}/final_model.keras")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
metrics_to_plot = [("loss", "Loss"), ("auc", "AUC"), ("recall", "Recall")]
for ax, (key, title) in zip(axes, metrics_to_plot):
    if key in history.history:
        ax.plot(history.history[key], label="train")
        ax.plot(history.history[f"val_{key}"], label="val")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/07_training_curves.png", dpi=120)
plt.close()

print(f"\nModel saved to: {MODEL_DIR}/best_model.keras")
print(f"Training curves saved to: {FIG_DIR}/07_training_curves.png")
print("\nStep 7 complete. Next: Step 8 (Evaluate Model on test set).")
