import numpy as np
import sys

# Implemented functions
from functions import saveNetwork, loadDataset, generateTarget

# # # # # # # # # # # #
# ---- Main code ---- #
# # # # # # # # # # # #

for i in range(36000):
  print(f"Current cycle: #{i}")
  X, Attacked_mat, mask = loadDataset(i)

  # Extract the node features to be attacked
  P = X[:, 0]
  Q = X[:, 1]

  # Apply the distribution-based attack and mark the attacked buses
  X[mask, 0] = np.random.normal(np.mean(P), np.std(P), size=mask.sum())
  X[mask, 1] = np.random.normal(np.mean(Q), np.std(Q), size=mask.sum())

  # Generate target (expected output)
  target = generateTarget( Attacked_mat, i, "d" )

  # Save the modified files
  saveNetwork(X, target, i, "d")



