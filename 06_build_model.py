"""
Step 6: Build LSTM Model (sanity check)
"""

import numpy as np
from lstm_model import build_lstm_model

X_train = np.load("processed_data/sequences/X_train.npy")
seq_len = X_train.shape[1]
n_features = X_train.shape[2]

print(f"Sequence length: {seq_len}")
print(f"Number of features: {n_features}")

model = build_lstm_model(seq_len, n_features)
model.summary()

total_params = model.count_params()
print(f"\nTotal trainable parameters: {total_params:,}")
print("\nStep 6 complete. Model architecture verified.")
print("Next: Step 7 (Train Model on all 40,336 patients).")
