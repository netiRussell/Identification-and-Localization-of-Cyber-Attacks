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
# Define initial mins and maxs
X0 = np.load(f"../init_dataset/x{0}.npy", mmap_mode="r")
P_min = X0[:, 0].min()
P_max = X0[:, 0].max()
Q_min = X0[:, 1].min()
Q_max = X0[:, 1].max()

# Get the final mins and maxs
for i in range(1,36000):
    print(f"Current graph: {i}")
    
    # Load a graph
    X = np.load(f"../init_dataset/x{i}.npy")
    
    # Extract the node features to be attacked
    p = X[:, 0]
    q = X[:, 1]

    # Update global mins/maxs
    cur_p_min, cur_p_max = p.min(), p.max()
    cur_q_min, cur_q_max = q.min(), q.max()

    if cur_p_min < P_min:
        P_min = cur_p_min
    if cur_p_max > P_max:
        P_max = cur_p_max
    if cur_q_min < Q_min:
        Q_min = cur_q_min
    if cur_q_max > Q_max:
        Q_max = cur_q_max


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
  target = np.concatenate([mask.astype(int), [1]])

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
  target = np.concatenate([mask.astype(int), [1]])

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
  target = np.concatenate([mask.astype(int), [0]])

  # Save the modified files
  saveNetwork(X, target, i)

