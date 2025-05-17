import numpy as np

# TODO: To be deleted
import sys

def saveNetwork( X, target, i, attack_type):
  """
  Saves the network into a Ad(or As)_dataset folder

  Parameters:
  ----------
  X : NumPy array filled with float values
    Node features array
    
  target: a NumPy array filled with integers
    Expected output of the model that is composed of the attack status and IDs of the attacked nodes

  i : Integer
    Current iteration (i from 0 to 36000)

  attack_type: String
    Type of the FDIA. 's' = data scale, 'd' = distribution-based

    
  Returns:
  -------
    None
  """
  np.save(f"../A{attack_type}_dataset/x{i}", X)
  np.save(f"../A{attack_type}_dataset/target{i}", target)



def loadDataset( i ):
  """
  Loads the network and randomly decides if the network will be attacked.
  Then, creates a boolean mask to attack random buses
  (mask has a size of 0 if the network will not experience any attack)

  Parameters:
  ----------
  i: Integer
    Current iteration (i from 0 to 36000)
  

  Returns:
  -------
  X : NumPy array filled with float values
    Node features array
    
  Attacked_mat: NumPy array filled with float values
    Matrix of nodes' attacked status

  mask : NumPy array filled with boolean values
    Boolean mask used to selectively modify only the rows of data corresponding to True values in it
  """
  # X = Node features array
  X = np.load(f"../init_dataset/x{i}.npy")

  # Attacked_mat = matrix of nodes' attacked status
  Attacked_mat = np.full(2848, False, dtype="bool")

  # With a 40% chance (60% chance the network won't experience any attack),
  # create a boolean mask to randomly attack up to 15 connected buses
  num_buses_tobe_attacked = 0
  mask = np.full(2848, False)

  if(np.random.rand() <= 0.4):
    # Randomly pick number of buses to be attacked (up to 15)
    num_buses_tobe_attacked = np.random.randint(1, 16)

    # Load direct neighbors of each node
    neighbors = np.load("../init_dataset/neighbors.npy", allow_pickle = True).tolist()
    
    # Randomly pick the root bus around which the attack will take place
    root = np.random.randint(0, 2848)

    # Find buses connected to the root with bfs 
    buses_tobe_attacked = _bfs(root, num_buses_tobe_attacked, neighbors)

    # Fill mask with the output of bfs
    mask[buses_tobe_attacked] = True


  # Set the attacked_flag
  Attacked_mat[mask] = True

  return X, Attacked_mat, mask



def generateTarget( Attacked_mat ):
  """
  Generates a NumPy array of integers that represents the
  expected output for the training of the model.
  [0] = no attack happended
  [1 , id1, id2, ...] = attack took place on the buses with the id1, id2, ... IDs

  Parameters:
  ----------
  Attacked_mat: NumPy array filled with float values
    Matrix of nodes' attacked status


  Returns:
  -------
    target: a NumPy array filled with integers
      Expected output of the model that is composed of the attack status and IDs of the attacked nodes

  """
  target = list()
  
  # -- Append IDs of the nodes that have been attacked --
  for node_id, node_attackedFlag in enumerate(Attacked_mat):
    if node_attackedFlag == True:
      target.append(node_id)
  
  # -- Create the final output --
  # If the list is empty => network was never attacked
  if( len(target) == 0 ):
    # expected output: healthy status of the network
    return np.array([0])
  else:
    # expected output: unhealthy status of the network + ids of the attacked buses
    target.insert(0, 1)
    return target


# # # # # # # # # # # # # # # #
# ---- Private functions ---- #
# # # # # # # # # # # # # # # #
from collections import deque

def _bfs(root, k, neighbors):
  visited = {root}
  q = deque([root])
  out = []

  while q and len(out) < k:
    u = q.popleft()

    for v in neighbors[u]:
      if v not in visited:
        visited.add(v)
        out.append(v)
        q.append(v)

        if len(out) >= k:
            break
        

  return out