import numpy as np

# Implemented functions
from functions import saveNetwork, loadDataset

# # # # # # # # # # # #
# ---- Main code ---- #
# # # # # # # # # # # #

for i in range(36000*2):
    
  print(f"Current cycle: #{i}")
  
  # -- As data subset --
  X, mask = loadDataset(i)

  # Apply the scale-based attack on the marked buses
  X[mask, 0] = X[mask, 0] * np.random.uniform(0.9, 1.1, size=mask.sum())
  X[mask, 1] = X[mask, 1] * np.random.uniform(0.9, 1.1, size=mask.sum())

  # Generate target (expected output)
  target = mask.astype(int)

  # Save the modified files
  saveNetwork(X, target, i )
  
  
  # -- Ad data subset --
  i+= 1
  X, mask = loadDataset(i)

  # Extract the node features to be attacked
  P = X[:, 0]
  Q = X[:, 1]

  # Apply the distribution-based attack on the marked buses
  X[mask, 0] = np.random.normal(np.mean(P), np.std(P), size=mask.sum())
  X[mask, 1] = np.random.normal(np.mean(Q), np.std(Q), size=mask.sum())

  # Generate target (expected output)
  target = mask.astype(int)

  # Save the modified files
  saveNetwork(X, target, i)




# TODO: Make all of these files to be saved at a single directory
# TODO: commit and push changes 