import numpy as np
import sys

# Implemented functions
from functions import saveAttackedNetwork, loadDataset

# # # # # # # # # # # #
# ---- Main code ---- #
# # # # # # # # # # # #

for i in range(36000):
  X, Attacked_mat, mask = loadDataset(i)

  # Extract the node features to be attacked
  P = X[:, 0]
  Q = X[:, 1]

  # Apply the distribution-based attack and mark the attacked buses
  X[mask, 0] = np.random.normal(np.mean(P), np.std(P), size=mask.sum())
  X[mask, 1] = np.random.normal(np.mean(Q), np.std(Q), size=mask.sum())

  # Save the modified files
  saveAttackedNetwork(X, Attacked_mat, i, "d")

  # TODO: implement and run the target generator function


