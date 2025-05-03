import numpy as np

def saveAttackedNetwork( X, Attacked_mat, i, attack_type):
  """
  Saves the attacked network into a Ad(or As)_dataset folder

  Parameters:
  ----------
  X : NumPy array filled with float values
    Node features array
    
  Attacked_mat: NumPy array filled with float values
    Matrix of nodes' attacked status

  i : Integer
    Current iteration (i from 0 to 36000)

  attack_type: String
    Type of the FDIA. 's' = data scale, 'd' = distribution-based

    
  Returns:
  -------
    None
  """
  np.save(f"../A{attack_type}_dataset/x{i}", X)
  np.save(f"../A{attack_type}_dataset/attacked_flag{i}", Attacked_mat)


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
  Attacked_mat = np.load(f"../init_dataset/attacked_flag{i}.npy")

  # With a 70% chance (30% chance the network won't experience any attack),
  # create a boolean mask to randomly attack up to 50% of the nodes
  percentage_tobe_attacked = 0
  if(np.random.rand() < 0.7):
    percentage_tobe_attacked = np.random.uniform(0, 0.5)

  mask = np.random.rand(X[:, 0].shape[0]) < percentage_tobe_attacked # the use of X[:, any].shape would yeild the same result

  # Set the attacked_flag
  Attacked_mat[mask] = True

  return X, Attacked_mat, mask



# TODO: target generation function to be used in Ad and As .py files to specify the expected output of a particular dataset
# def generateTarget( Attacked_mat ):