import numpy as np
import torch

# Implemented functions
from functions import saveNetwork, loadDataset

# TODO: to be deleteds
import sys

# # # # # # # # # # # #
# ---- Main code ---- #
# # # # # # # # # # # #

dataset_config = {
    "normal_scaling": False
    }


"""
# --- Compare Ad vs As influence ---
X0, mask = loadDataset(100, attack=True)
Xd = np.copy(X0)
    
# Extract all node features
P = Xd[:, 0]
Q = Xd[:, 1]

# Don't follow outliners, so that the attack is not as easy to detect
# Get mean and STD for 90% of P
lo, hi = np.percentile(P, [5,95])
P_trim = P[(P>=lo)&(P<=hi)]
P_mu, P_std  = P_trim.mean(), P_trim.std() / 4

# Get mean and STD for 90% of Q
lo, hi = np.percentile(Q, [5,95])
Q_trim = Q[(Q>=lo)&(Q<=hi)]
Q_mu, Q_std  = Q_trim.mean(), Q_trim.std() / 4

# Attack buses that have some load only (otherwise => easy to detect)
mask_P = mask & (P != 0)
mask_Q = mask & (Q != 0)

# Apply the distribution-based attack on the marked buses
Xd[mask_P, 0] = np.random.normal(P_mu, P_std, size=mask_P.sum())
Xd[mask_Q, 1] = np.random.normal(Q_mu, Q_std, size=mask_Q.sum())

print(f"Xd: {np.mean(np.abs(X0-Xd))}")


Xs = np.copy(X0)
Xs[mask, 0] = Xs[mask, 0] * np.random.uniform(0.9, 1.1, size=mask.sum())
Xs[mask, 1] = Xs[mask, 1] * np.random.uniform(0.9, 1.1, size=mask.sum())
print(f"Xs: {np.mean(np.abs(X0-Xs))}")

sys.exit()
"""



if( dataset_config["normal_scaling"] ):
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
      
  # Extract all node features
  P = X[:, 0]
  Q = X[:, 1]

  # Don't follow outliners, so that the attack is not as easy to detect
  # Get mean and STD for 90% of P
  lo, hi = np.percentile(P, [5,95])
  P_trim = P[(P>=lo)&(P<=hi)]
  P_mu, P_std  = P_trim.mean(), P_trim.std() / 4
  
  # Get mean and STD for 90% of Q
  lo, hi = np.percentile(Q, [5,95])
  Q_trim = Q[(Q>=lo)&(Q<=hi)]
  Q_mu, Q_std  = Q_trim.mean(), Q_trim.std() / 4
  
  # Attack buses that have some load only (otherwise => easy to detect)
  mask_P = mask & (P != 0)
  mask_Q = mask & (Q != 0)

  # Apply the distribution-based attack on the marked buses
  X[mask_P, 0] = np.random.normal(P_mu, P_std, size=mask_P.sum())
  X[mask_Q, 1] = np.random.normal(Q_mu, Q_std, size=mask_Q.sum())
  
  # Exclude Voltage magnitude and angle
  X = X[:, :2]
  
  if( dataset_config["normal_scaling"] ):
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
  
  if( dataset_config["normal_scaling"] ):
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
  
  if( dataset_config["normal_scaling"] ):
      # Apply normal scaling [0,1]
      X[:,0] = (X[:,0] - P_min) / (P_max - P_min + 1e-8)
      X[:,1] = (X[:,1] - Q_min) / (Q_max - Q_min + 1e-8)

  # Generate target(expected output) for multi-label supervised learning
  target = np.concatenate([mask.astype(int), [0]])

  # Save the modified files
  saveNetwork(X, target, i)

