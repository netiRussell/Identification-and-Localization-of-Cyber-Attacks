import numpy as np

# Implemented functions
from functions import saveNetwork, loadDataset

import sys

# # # # # # # # # # # #
# ---- Main code ---- #
# # # # # # # # # # # #

for i in range(36000):
    
  print(f"\nCurrent cycle: #{i}")
  
  # Load a node features and a mask with the buses to be attacked
  X, mask = loadDataset(i)
  
  if( np.random.rand() < 0.5 ):
      # -- As data subset --
      print("As attack chosen")
    
      # Apply the scale-based attack on the marked buses
      X[mask, 0] = X[mask, 0] * np.random.uniform(0.9, 1.1, size=mask.sum())
      X[mask, 1] = X[mask, 1] * np.random.uniform(0.9, 1.1, size=mask.sum())
      
  else :
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

  # Generate target (expected output)
  target = mask.astype(int)

  # Save the modified files
  saveNetwork(X, target, i)

