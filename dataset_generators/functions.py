import numpy as np
import os
import torch
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib
import random
from torch_geometric.loader import DataLoader

# TODO: To be deleted
import sys
# Enable reproducibility
torch.backends.cudnn.deterministic = True
random.seed(123)
torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)


def saveNetwork( Z, target, i):
  """
  Saves the network into a Ad(or As)_dataset folder

  Parameters:
  ----------
  Z : NumPy array filled with float values
    Node features array
    
  target: a NumPy array filled with integers
    Expected output of the model that is composed of the attack status and IDs of the attacked nodes

  i : Integer
    Current iteration (i from 0 to 36000)

    
  Returns:
  -------
    None
  """
  np.save(f"../final_dataset/Z_{i}", Z)
  np.save(f"../final_dataset/target_{i}", target)



def loadDataset( i, attack=False, number_nodes=300 ):
  """
  Loads the network and randomly decides if the network will be attacked.
  Then, creates a boolean mask to attack random buses
  (mask has a size of 0 if the network will not experience any attack)

  Parameters:
  ----------
  i: Integer
    Current iteration
    
  attack: Boolean
      Boolean that represents whether this sample will be attacked
  

  Returns:
  -------
  Z : NumPy array filled with float values
    Node features array

  mask : NumPy array filled with boolean values
    Boolean mask used to selectively modify only the rows of data corresponding to True values in it
  """
  # X = Node features array
  Z = np.load(f"../init_dataset/Z_{i}.npy")

  # Declare and Define with default values mask and num of buses to be attacked
  num_buses_tobe_attacked = 0
  mask = np.full(number_nodes, False, dtype="bool")

  if(attack == True):
    print("There will be an attack")
    
    # Randomly pick number of buses to be attacked between 5-10% of the buses
    num_buses_tobe_attacked = int(np.random.uniform(0.05, 0.10) * number_nodes)

    # Load direct neighbors of each node
    neighbors = np.load("../init_dataset/neighbors.npy", allow_pickle = True).tolist()
    
    # Randomly pick the root bus around which the attack will take place
    root = np.random.randint(0, number_nodes)

    # Find buses connected to the root with bfs 
    buses_tobe_attacked = _bfs(root, num_buses_tobe_attacked, neighbors)

    # Fill mask with the output of bfs
    mask[buses_tobe_attacked] = True


  return Z, mask


def save_checkpoint(state, path='./saved_grads/'):
    """
    Saves current progress as a .tar file

    Parameters:
    ----------
    state : a Python object
      Object containing model's and optimizer's parameters
      
    path: a string
      Path to the folder where the file containing current state will be saved

      
    Returns:
    -------
      None
    """
    # Get current date and time
    now = datetime.now()
    timestamp = now.strftime("%Y_%m_%d")
    
    # Define the final path with file name
    finalPath = path + "checkpoint" + timestamp + ".pth.tar"
    
    """
    # Make sure not to overwrite previous saved grads
    counter = 0
    while(os.path.isfile(finalPath)):
        finalPath = finalPath = path + "checkpoint" + counter + timestamp + ".pth.tar"
        counter += 1
    """
    
    torch.save(state, finalPath)



def visualizeLossValid(fa, dr, f1, acc):
    """
    Visualizes precision, recall, f1, and accuracy collected during the training
    

    Parameters:
    ----------
    fa: a NumPy array
      An array that contains fa values from each epoch of the training
      
    dr: a NumPy array
      An array that contains dr values from each epoch of the training
    
    f1: a NumPy array
      An array that contains F1 values from each epoch of the training
    
    acc: a NumPy array
      An array that contains accuracy values from each epoch of the training

      
    Returns:
    -------
      None
    """
    # Create a figure with two subplots side by side
    fig, ((g1, g2), (g3, g4)) = plt.subplots(2, 2)
    
    # First plot
    g1.plot(range(1,len(fa)+1), fa, label='FA')
    g1.set_title('FA over epoch')
    g1.set_xlabel('Epoch #')
    g1.set_ylabel('FA')
    g1.legend()
    
    # Second plot
    g2.plot(range(1, len(dr)+1), dr, label='DR')
    g2.set_title('DR over epoch')
    g2.set_xlabel('Epoch #')
    g2.set_ylabel('DR')
    g2.legend()
    
    # Third plot
    g3.plot(range(1, len(f1)+1), f1, label='F1')
    g3.set_title('F1 over epoch')
    g3.set_xlabel('Epoch #')
    g3.set_ylabel('F1')
    g3.legend()
    
    # Third plot
    g4.plot(range(1, len(acc)+1), acc, label='Accuracy')
    g4.set_title('Accuracy over epoch')
    g4.set_xlabel('Epoch #')
    g4.set_ylabel('Accuracy')
    g4.legend()
    
    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()
    

def preparePyGDataset(config, FDIADataset):
    """
    Prepares PyG loaders and computes the demensions of input features
    

    Parameters:
    ----------
    config: a Python dictionary
      A dictionary of settings for the model and datasets
      
    FDIADataset: a PyTorch Dataset class
      A PyG dataset that turns NumPy arrays into structured 'Data's

      
    Returns:
    -------
      train_loader: a PyTorch DataLoader
          A DataLoader that holds samples for training
          
      valid_loader: a PyTorch DataLoader
          A DataLoader that holds samples for validation
          
      test_loader: a PyTorch DataLoader
          A DataLoader that holds samples for testing
          
      in_feats: Integer
          An integer that represents the demensions of the input for the 1st ChebConv layer
          Which is just 4=(P, Q, V, angle) or 2=(P, Q) per node
    """
    
    # Definition of the lists containing all indices
    all_indices = list(range(34561))

    # Get training indices
    train_indices = all_indices[1:23041]

    # Get validation indices
    val_indices = all_indices[23041:28801]
    
    # Get test indices
    test_indices = all_indices[28801:34561]

    # Mix attacks and normal indices within the subsets
    random.shuffle(train_indices)
    random.shuffle(val_indices)
    # ! Test is not shuffled for easier debugging

    # Get datasets
    train_dataset = FDIADataset(train_indices, config["dataset_root"])
    val_dataset = FDIADataset(val_indices, config["dataset_root"])
    test_dataset = FDIADataset(test_indices, config["dataset_root"])

    # Define loaders for each data subset to sequentially load batches
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    valid_loader   = DataLoader(val_dataset,   batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1)

    # Get input features (Size of X in the first dimension)
    # Supposed to be 4 (P, Q, V, angle) or 2 (P, Q)
    in_feats = train_dataset[0].x.size(1)
    
    return train_loader, valid_loader, test_loader, in_feats


def selectDevice():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    print(f"Device selected: {device}")
    
    return device


def generateDatasetStandardScaler( indices ):
    """
    Generates data scaler and saves it on a disk to be utilized in __getitem__
    

    Parameters:
    ----------
    indices: a Python list
        List of sample indices that will be used in the training set

      
    Returns:
    -------
      None
      
    Comments:
    -------
    To use, in dataset.py, in init function, add:
        # Load the fitted StandardScaler:
        scaler = joblib.load("train_scaler.joblib")
        
        # Save mean and scale from the scaler as model's parameters
        # scaler.mean_ has a shape of (num_node_feats,), i.e. (P,Q or P,Q,V,angle)
        # scaler.scale_ has a shape of (num_node_feats,), i.e. (2 or 4)
        self.register_buffer("mean_", torch.tensor(scaler.mean_, dtype=torch.float))
        self.register_buffer("scale_", torch.tensor(scaler.scale_, dtype=torch.float))
    Then, in __getitem__, add:
        # Apply Standard scaling
        x = (x - self.mean_) / self.scale_
        
    """
    print("Dataset Standard scaler generation is in progres...")
    
    # x_i.npy has shape (2848, num_feats),
    # The 2D array to be accumulated (num_train * 2848, num_feats).
    all_nodes = []   
    for i in indices:
        Xi = np.load(f"./dataset/x_{i}.npy")   # shape=(2848,2 or 4)
        all_nodes.append(Xi)
        
    # Stack into shape (num_train, 2848, 2 or 4):
    all_nodes = np.stack(all_nodes, axis=0)
    
    # Reshape to (num_train * 2848, 2 or 4):
    ntot = all_nodes.shape[0] * all_nodes.shape[1]
    all_nodes = all_nodes.reshape(ntot, all_nodes.shape[2])
    
    # Fit the scaler on these flattened features:
    scaler = StandardScaler().fit(all_nodes)  
    
    # Save the scaler to disk (you can also just save mean_/scale_ arrays):
    joblib.dump(scaler, "train_scaler.joblib")


# # # # # # # # # # # # # # # #
# ---- Private functions ---- #
# # # # # # # # # # # # # # # #
from collections import deque

"""
Runs vanilla BFS algorithm
"""
def _bfs(root, k, neighbors):
  visited = {root}
  q = deque([root])
  out = [root]

  while q and len(out) < k:
    u = q.popleft()

    for v in neighbors[u]:
      if v not in visited:
        visited.add(v)
        out.append(v)
        q.append(v)

        if len(out) >= k:
            break
        
  print(f"# of buses attacked: {len(out)}")
  return out