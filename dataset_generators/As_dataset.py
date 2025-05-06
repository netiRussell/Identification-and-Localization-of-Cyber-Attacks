import numpy as np

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

  # Apply the scale-based attack on the marked buses
  X[mask, 0] = X[mask, 0] * np.random.uniform(0.9, 1.1, size=mask.sum())
  X[mask, 1] = X[mask, 1] * np.random.uniform(0.9, 1.1, size=mask.sum())

  # Generate target (expected output)
  target = generateTarget( Attacked_mat, i, "s" )

  # Save the modified files
  saveNetwork(X, target, i, "s")



