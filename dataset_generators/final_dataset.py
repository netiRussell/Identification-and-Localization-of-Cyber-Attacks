import numpy as np

# Implemented functions
from functions import saveNetwork, loadDataset

import sys
import torch

# # # # # # # # # # # #
# ---- Main code ---- #
# # # # # # # # # # # #

# The first half, attacked with Ad
for i in range(9000):
  print(f"\nCurrent cycle: #{i}")
  
  # Load a node features and a mask with the buses to be attacked marked as True
  X, mask = loadDataset(i, attack=True)

  # -- Ad data subset --
  print("Ad attack chosen")
      
  # Extract the node features to be attacked
  P = X[:, 0]
  Q = X[:, 1]

  # Apply the distribution-based attack on the marked buses
  X[mask, 0] = np.random.normal(np.mean(P), np.std(P), size=mask.sum())
  X[mask, 1] = np.random.normal(np.mean(Q), np.std(Q), size=mask.sum())
  
  # Exclude Voltage magnitude and angle
  X = X[:, :2]
  
  # Normalize the sample:
  # Get a range of min and max values
  min_vals = np.min(X, axis=0)
  max_vals = np.max(X, axis=0)
  range_vals = np.clip(max_vals - min_vals, a_min=1e-8, a_max=None)
  # Apply normal scaling [0,1]
  X = (X - min_vals) / range_vals

  # Generate target (expected output)
  target = mask.astype(int)

  # Save the modified files
  saveNetwork(X, target, i)

# The first half, attacked with As
for i in range(9000, 18000):
  print(f"\nCurrent cycle: #{i}")
  
  # Load a node features and a mask with the buses to be attacked marked as True
  X, mask = loadDataset(i, attack=True)

  # -- As data subset --
  print("As attack chosen")
    
  # Apply the scale-based attack on the marked buses
  X[mask, 0] = X[mask, 0] * np.random.uniform(0.9, 1.1, size=mask.sum())
  X[mask, 1] = X[mask, 1] * np.random.uniform(0.9, 1.1, size=mask.sum())
  
  # Exclude Voltage magnitude and angle
  X = X[:, :2]
  
  # Normalize the sample:
  # Get a range of min and max values
  min_vals = np.min(X, axis=0)
  max_vals = np.max(X, axis=0)
  range_vals = np.clip(max_vals - min_vals, a_min=1e-8, a_max=None)
  # Apply normal scaling [0,1]
  X = (X - min_vals) / range_vals

  # Generate target (expected output)
  target = mask.astype(int)

  # Save the modified files
  saveNetwork(X, target, i)

 # The second half, no attack 
for i in range(18000, 36000):
  print(f"\nCurrent cycle: #{i}")
  
  # Load a node features and a mask filled with False values
  X, mask = loadDataset(i, attack=False)
  
  # Exclude Voltage magnitude and angle
  X = X[:, :2]
  
  # Normalize the sample:
  # Get a range of min and max values
  min_vals = np.min(X, axis=0)
  max_vals = np.max(X, axis=0)
  range_vals = np.clip(max_vals - min_vals, a_min=1e-8, a_max=None)
  # Apply normal scaling [0,1]
  X = (X - min_vals) / range_vals

  # Generate target (expected output)
  target = mask.astype(int)

  # Save the modified files
  saveNetwork(X, target, i)

