import numpy as np
import torch

# Implemented functions
from functions import saveNetwork, loadDataset

# TODO: to be deleteds
import sys

# # # # # # # # # # # #
# ---- Main code ---- #
# # # # # # # # # # # #

# -- Prepare normal scaling --
# Collect all P and Q
all_P = np.load(f"../init_dataset/x{0}.npy")[:, 0]
all_Q = np.load(f"../init_dataset/x{0}.npy")[:, 1]

for i in range(1,36000):
    print(f"Current graph: {i}")
    
    # Load a graph
    X = np.load(f"../init_dataset/x{i}.npy")
    
    # Extract the node features to be attacked
    all_P = np.concatenate([all_P, X[:, 0]])
    all_Q = np.concatenate([all_Q, X[:, 1]])

# Define mins and maxs
P_min, P_max = all_P.min(), all_P.max()
Q_min, Q_max = all_Q.min(), all_Q.max()
    

# -- The first half, attacked with Ad --
for i in range(9000):
  print(f"\nCurrent cycle: #{i}")
  
  # Load a node features and a mask with the buses to be attacked marked as True
  X, mask = loadDataset(i, attack=True)

  # - Ad data subset -
  print("Ad attack chosen")
      
  # Extract the node features to be attacked
  P = X[:, 0]
  Q = X[:, 1]

  # Apply the distribution-based attack on the marked buses
  X[mask, 0] = np.random.normal(np.mean(P), np.std(P), size=mask.sum())
  X[mask, 1] = np.random.normal(np.mean(Q), np.std(Q), size=mask.sum())
  
  # Exclude Voltage magnitude and angle
  X = X[:, :2]
  
  # Apply normal scaling [0,1]
  X[:,0] = (X[:,0] - P_min) / (P_max - P_min + 1e-8)
  X[:,1] = (X[:,1] - Q_min) / (Q_max - Q_min + 1e-8)

  # Generate target(expected output) for multi-label supervised learning
  target = np.array((1,2))
  target[0] = mask.astype(int) # node-level
  target[1] = np.array([1]) # graph-level

  # Save the modified files
  saveNetwork(X, target, i)

# -- The first half, attacked with As -- 
for i in range(9000, 18000):
  print(f"\nCurrent cycle: #{i}")
  
  # Load a node features and a mask with the buses to be attacked marked as True
  X, mask = loadDataset(i, attack=True)

  # - As data subset -
  print("As attack chosen")
    
  # Apply the scale-based attack on the marked buses
  X[mask, 0] = X[mask, 0] * np.random.uniform(0.9, 1.1, size=mask.sum())
  X[mask, 1] = X[mask, 1] * np.random.uniform(0.9, 1.1, size=mask.sum())
  
  # Exclude Voltage magnitude and angle
  X = X[:, :2]
  
  # Apply normal scaling [0,1]
  X[:,0] = (X[:,0] - P_min) / (P_max - P_min + 1e-8)
  X[:,1] = (X[:,1] - Q_min) / (Q_max - Q_min + 1e-8)

  # Generate target(expected output) for multi-label supervised learning
  target = np.array((1,2))
  target[0] = mask.astype(int) # node-level
  target[1] = np.array([1]) # graph-level

  # Save the modified files
  saveNetwork(X, target, i)

 # -- The second half, no attack --
for i in range(18000, 36000):
  print(f"\nCurrent cycle: #{i}")
  
  # Load a node features and a mask filled with False values
  X, mask = loadDataset(i, attack=False)
  
  # Exclude Voltage magnitude and angle
  X = X[:, :2]
  
  # Apply normal scaling [0,1]
  X[:,0] = (X[:,0] - P_min) / (P_max - P_min + 1e-8)
  X[:,1] = (X[:,1] - Q_min) / (Q_max - Q_min + 1e-8)

  # Generate target(expected output) for multi-label supervised learning
  target = np.array((1,2))
  target[0] = mask.astype(int) # node-level
  target[1] = np.array([0]) # graph-level

  # Save the modified files
  saveNetwork(X, target, i)

