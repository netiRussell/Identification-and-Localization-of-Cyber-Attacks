import numpy as np
import os
import torch
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib
import random
from torch_geometric.loader import DataLoader
import pandapower.topology as pptop
import networkx as nx
from torch.nn.utils import clip_grad_norm_

'''
from model.ARMA import GNNArma
from model.ARMA_Transformer import GNNArmaTransformer
from model.CGCN import CGCN
from model.CNN import CNN
from model.MLP import MLP
from model.LSTM import LSTM
'''

# TODO: To be deleted
import sys
# Enable reproducibility
torch.backends.cudnn.deterministic = True
random.seed(123)
torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)


def saveNetwork( X, target, i, location):
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

  location : String
    Path to where the target and X will be saved

    
  Returns:
  -------
    None
  """
  np.save(f"{location}/X_{i}", X)
  np.save(f"{location}/target_{i}", target)


def load_tobeAttacked(G, irrelevant_buses, t_load, entry_bus, radius, number_nodes=300):
    """
    Returns node features of timestep t_load and a mask 
    filled with False values of size number_nodes

    Parameters:
    ----------
    G : NX network object
       Graph topology of the currently chosen electrical network(bus system)

    irrelevant_buses : list or set of integers
        Buses that should be excluded from attack consideration (e.g., generator or zero-injection buses).
 
    t_load : Integer
        Current timestep used to load the correct snapshot of node features.
 
    entry_bus : Integer
        The bus from which the attack (or BFS traversal) begins.
 
    radius : Integer
        The maximum number of hops from the entry bus to include in the BFS region.

    Returns:
    -------
    Z : NumPy array filled with float values
      Node features array

    mask : NumPy array filled with boolean values
      Boolean mask used to selectively modify only the rows of data corresponding to True values in it
    
    T_x: list of integers
        List of nodes to be attacked
    """
    T_x = list()
    
    # Make sure the number of attacked buses are always >= 2 (2 for IEEE14, 5 for IEEE300)
    while(len(T_x) < 2):
        lengths = nx.single_source_shortest_path_length(G, entry_bus, cutoff=radius)
        attacked_buses = list(lengths.keys())
    
        # Filter out the 0-injection and generator buses (otherwise => easy to detect)
        T_x = [b for b in attacked_buses if b not in irrelevant_buses]
    
    # Create an empty mask
    mask = np.full(number_nodes, False, dtype="bool")
    
    # Fill the mask with the buses to be attacked
    mask[T_x] = True
    
    # Z = Node features array
    Z = np.load(f"../init_dataset/Z_{t_load}.npy")
    
    return Z, mask, T_x

def load_honest( t_load, number_nodes=300 ):
  """
  Returns node features of timestep t_load and a mask 
  filled with False values of size number_nodes

  Parameters:
  ----------
  t_load: Integer
    Current timestep
    
  number_nodes: Integer
      Number of nodes present in the current network
      

  Returns:
  -------
  Z : NumPy array filled with float values
    Node features array

  mask : NumPy array filled with boolean values
    Boolean mask used to selectively modify only the rows of data corresponding to True values in it
  """
  # Z = Node features array
  Z = np.load(f"../init_dataset/Z_{t_load}.npy")

  # Declare and Define mask with no True values(nothing to attack)
  mask = np.full(number_nodes, False, dtype="bool")

  return Z, mask


def save_checkpoint(state, name="checkpoint", path='./saved_grads/'):
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
    finalPath = path + name + timestamp + ".tar"
    
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
    
    print(fa)
    
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
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, pin_memory=True )
    valid_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

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


# -- Definitions of the models --
def define_modelCGCN(hyperparams, config):
    
    if hyperparams == None:
        model = CGCN(
            in_channels=config["in_channels"],
            num_CGCN_layers=config["cgcn_specific"]["num_CGCN_layers"],
            u=config["cgcn_specific"]["u"],
            Ks=config["cgcn_specific"]["Ks"],
            dropout=config["cgcn_specific"]["dropout"],
            num_nodes = config["num_nodes"],
        )
    else:
        # Required hyperparams content: num_CGCN_layers, dropout, u, Ks
        model = CGCN(
            in_channels=config["in_channels"],
            num_CGCN_layers=hyperparams["num_CGCN_layers"],
            u=hyperparams["u"],
            Ks=hyperparams["Ks"],
            dropout=hyperparams["dropout"],
            num_nodes = config["num_nodes"],
        )
    
    return model

def define_modelARMA(hyperparams, config):
    if hyperparams == None:
        model = GNNArma(
            in_channels=config["in_channels"], 
            num_ARMA_layers=config["arma_specific"]["num_ARMA_layers"],
            hidden_channels=config["arma_specific"]["hidden_channels"],
            num_stacks=config["arma_specific"]["num_stacks"],
            num_layers=config["arma_specific"]["num_layers"], 
            dropout=config["arma_specific"]["dropout"],
            num_nodes=config["num_nodes"]
        )
    else:
        # Required hyperparams content: dropout, num_ARMA_layers, hidden_channels, num_stacks, num_layers
        model = GNNArma(
            in_channels=config["in_channels"], 
            num_ARMA_layers=hyperparams["num_ARMA_layers"],
            hidden_channels=hyperparams["hidden_channels"],
            num_stacks=hyperparams["num_stacks"],
            num_layers=hyperparams["num_layers"], 
            dropout=hyperparams["dropout"],
            num_nodes=config["num_nodes"]
        )
    
    return model

def define_modelTransformer(hyperparams, config):
    if hyperparams == None:
        model = GNNArmaTransformer(
            in_channels=config["in_channels"],
            num_ARMA_layers=config["transformer_specific"]["num_ARMA_layers"],
            hidden_channels=config["transformer_specific"]["hidden_channels"],
            d_model_channels=config["transformer_specific"]["d_model_channels"],
            dim_feedforward = config["transformer_specific"]["dim_feedforward"],
            num_stacks=config["transformer_specific"]["num_stacks"], 
            num_layers=config["transformer_specific"]["num_layers"],
            transformer_heads=config["transformer_specific"]["transformer_heads"],
            transformer_layers=config["transformer_specific"]["transformer_layers"],
            dropout=config["transformer_specific"]["dropout"],
            num_nodes=config["num_nodes"],
            leakyRelu_rate=config["transformer_specific"]["leakyRelu_rate"]
        )
    else:
        model = GNNArmaTransformer(
            in_channels=config["in_channels"],
            num_ARMA_layers=hyperparams["num_ARMA_layers"],
            hidden_channels=hyperparams["hidden_channels"],
            d_model_channels=hyperparams["d_model_channels"],
            dim_feedforward = hyperparams["dim_feedforward"],
            num_stacks=hyperparams["num_stacks"], 
            num_layers=hyperparams["num_layers"],
            transformer_heads=config["transformer_specific"]["transformer_heads"],
            transformer_layers=hyperparams["transformer_layers"],
            dropout=hyperparams["dropout"],
            num_nodes=config["num_nodes"],
            leakyRelu_rate=config["transformer_specific"]["leakyRelu_rate"]
        )
    
    return model

def define_modelCNN(hyperparams, config):
    if hyperparams == None:
        model = CNN(
            in_channels=config["in_channels"], 
            num_CNN_layers=config["cnn_specific"]["num_CNN_layers"],
            Ks=config["cnn_specific"]["Ks"],
            units=config["cnn_specific"]["units"],
            padding=config["cnn_specific"]["padding"],
            dropout=config["cnn_specific"]["dropout"],
            num_nodes=config["num_nodes"]
        )
    else:
        # Required hyperparams content: dropout, num_ARMA_layers, hidden_channels, num_stacks, num_layers
        model = CNN(
            in_channels=config["in_channels"], 
            num_CNN_layers=hyperparams["num_CNN_layers"],
            Ks=hyperparams["Ks"],
            units=hyperparams["units"],
            padding=hyperparams["padding"],
            dropout=hyperparams["dropout"],
            num_nodes=config["num_nodes"]
        )
    
    return model

def define_modelMLP(hyperparams, config):
    if hyperparams == None:
        model = MLP(
            in_channels=config["in_channels"], 
            num_MLP_layers=config["mlp_specific"]["num_MLP_layers"],
            units=config["mlp_specific"]["units"],
            dropout=config["mlp_specific"]["dropout"],
            num_nodes=config["num_nodes"]
        )
    else:
        # Required hyperparams content: dropout, num_ARMA_layers, hidden_channels, num_stacks, num_layers
        model = MLP(
            in_channels=config["in_channels"], 
            num_MLP_layers=hyperparams["num_MLP_layers"],
            units=hyperparams["units"],
            dropout=hyperparams["dropout"],
            num_nodes=config["num_nodes"]
        )
    
    return model

def define_modelLSTM(hyperparams, config):
    if hyperparams == None:
        model = LSTM(
            in_channels=config["in_channels"], 
            num_LSTM_layers=config["lstm_specific"]["num_LSTM_layers"],
            units=config["lstm_specific"]["units"],
            dropout=config["lstm_specific"]["dropout"],
            num_nodes=config["num_nodes"]
        )
    else:
        model = LSTM(
            in_channels=config["in_channels"], 
            num_LSTM_layers=hyperparams["num_LSTM_layers"],
            units=hyperparams["units"],
            dropout=hyperparams["dropout"],
            num_nodes=config["num_nodes"]
        )
        
    return model

# -- Training & evaluation functions --
def train_epoch(model, optimizer, epoch, train_loader, criterion, 
                scheduler, config, device, autocast, use_amp, scaler):
    model.train()

    # derive accumulation from config
    accum_steps = max(1, config["accumulated_batch_size"] // config["batch_size"])

    total_loss = torch.zeros((), device=device)  # stays on GPU

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(train_loader):
        batch = batch.to(device, non_blocking=True)

        with autocast(device_type=device.type if device.type!='cpu' else 'cuda',
                      enabled=use_amp):
            # logits
            logits_nodes, logits_graph = model(
                batch.x, batch.edge_index, weights=batch.edge_attr, batch=batch.batch
            )
            # normalize shapes
            logits_nodes = logits_nodes.reshape(-1)
            targets_nodes = batch.y.to(logits_nodes.dtype)

            # raw loss for logging
            raw_loss = criterion(logits_nodes, targets_nodes)

        # accumulate loss for logging
        total_loss += raw_loss.detach()

        # divide only for backward to average gradients across accum_steps
        loss_for_backward = raw_loss / accum_steps

        scaler.scale(loss_for_backward).backward()
        
        # Full-batch step
        if ((step + 1) % accum_steps == 0) or ((step + 1) == len(train_loader)):
            if use_amp:
                scaler.unscale_(optimizer)  # <-- needed before clipping
            
            # Check one what is the current grad (Commented out for performance)
            # Clip spikes in grad
            clip_grad_norm_(model.parameters(), max_norm=10.0)  
            '''
            total_norm = clip_grad_norm_(model.parameters(), max_norm=10.0)
            print(f"Grad norm before clipping: {total_norm:.4f}")
            '''
            
            # Print current epoch, batch, loss (Commented out for performance)
            '''
            print(f"Epoch#{epoch}, Batch#{(step // accum_steps):04d} | Current loss: {loss_for_backward:.4f}\n")
            '''
            
            # Applying grads
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

    # avg per minibatch
    return (total_loss / len(train_loader)).item()


def train_epoch_old(model, optimizer, epoch, train_loader, criterion, 
                scheduler, config, device, autocast, use_amp, scaler):
    # TO BE DELETED
    gradSpike = False
    
    # Turn on the training mode to include features like GraphNorm normalization
    model.train()
    
    # Declare default value for loss and # of batches(steps) for grad accumulation
    total_loss = torch.zeros(()).to(device)
    accum_steps = 64 # 64 mini batches filled with 4 samples = 256 samples per optimizer.step()
    
    # Reset accumulated grads
    optimizer.zero_grad()
    
    for minibatch_id, minibatch in enumerate(train_loader):
        minibatch = minibatch.to(device)
        
        with autocast(device_type=device.type if device.type!='cpu' else 'cuda',
                      enabled=use_amp):
            # Get targets for the batch
            target_nodes = minibatch.y
            target_graph = minibatch.y_graph
            
            # Get model's raw output (logits)
            logits_nodes, logits_graph = model(minibatch.x, 
                           minibatch.edge_index, 
                           weights=minibatch.edge_attr,
                           batch=minibatch.batch)
            
            # Compute loss and save it
            loss = criterion(logits_nodes.view(-1), target_nodes)
            loss += (criterion(logits_graph, target_graph.squeeze(0)))
            total_loss += (loss.item()/accum_steps)

        # Scale down the loss so that the grads are averaged over accum_steps
        scaler.scale(loss / accum_steps).backward()
        
        # Gradient Accumulation to simulate 2^8 samples per batch while maintaining low RAM usage
        if (minibatch_id+1) % accum_steps == 0:
            # Clip gradients so their total L2 norm ≤ 1
            clip_grad_norm_(model.parameters(), max_norm=10.0)
            
            # Applying grads
            # print(f"Epoch#{epoch}, Batch#{(minibatch_id // accum_steps):04d} | Current loss: {loss:.4f}\n")
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
            optimizer.zero_grad()
    
    # Extra gradient step to make sure there are no leftovers
    if (minibatch_id + 1) % accum_steps != 0:
        # Clip gradients so their total L2 norm ≤ 1
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Applying grads
        # print(f"Epoch#{epoch}, Last batch\n")
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()
        
    return (total_loss / len(train_loader)).item()


def validate(model, valid_loader, criterion, config, device):
    model.eval()

    eps = 1e-12 # needed for efficient prevention of the division by zero
    n_batches = len(valid_loader)
    num_nodes = config["num_nodes"]

    total_loss = torch.zeros((), device=device)

    # SW counters
    F1_95_SW = torch.zeros((), device=device, dtype=torch.long)
    F1_5_SW  = torch.zeros((), device=device, dtype=torch.long)
    localization_acc = torch.zeros((), device=device, dtype=torch.long)
    detection_acc    = torch.zeros((), device=device, dtype=torch.long)

    # NW streaming confusion per bus
    TP_b = torch.zeros(num_nodes, device=device, dtype=torch.long)
    FP_b = torch.zeros(num_nodes, device=device, dtype=torch.long)
    FN_b = torch.zeros(num_nodes, device=device, dtype=torch.long)

    # Graph-level confusion
    gTP = torch.zeros((), device=device, dtype=torch.long)
    gFP = torch.zeros((), device=device, dtype=torch.long)
    gFN = torch.zeros((), device=device, dtype=torch.long)
    gTN = torch.zeros((), device=device, dtype=torch.long)

    with torch.inference_mode():
        for batch in valid_loader:
            batch = batch.to(device, non_blocking=True)

            logits_nodes, logits_graph = model(
                batch.x, batch.edge_index, weights=batch.edge_attr, batch=batch.batch
            )
            logits_nodes = logits_nodes.reshape(-1)   # [num_nodes]
            logits_graph = logits_graph.reshape(-1)   # [B] (usually 1)

            targets_nodes = batch.y.to(logits_nodes.dtype)      # float
            targets_graph = batch.y_graph.reshape(-1).to(torch.bool)

            total_loss += criterion(logits_nodes, targets_nodes)

            # Threshold logits directly (BCEWithLogitsLoss)
            pred_nodes = (logits_nodes > 0)                      # bool [num_nodes]
            pred_graph = (logits_graph > 0)                      # bool [B]

            # Graph metrics
            detection_acc += (pred_graph == targets_graph).all().long()
            gTP += (pred_graph &  targets_graph).sum()
            gFP += (pred_graph & ~targets_graph).sum()
            gFN += (~pred_graph &  targets_graph).sum()
            gTN += (~pred_graph & ~targets_graph).sum()

            # Sample-wise SW metrics
            t_nodes_b = targets_nodes.to(torch.bool)
            TP = (pred_nodes &  t_nodes_b).sum()
            FP = (pred_nodes & ~t_nodes_b).sum()
            FN = (~pred_nodes &  t_nodes_b).sum()

            precision = TP.float() / (TP.float() + FP.float() + eps)
            recall    = TP.float() / (TP.float() + FN.float() + eps)
            f1        = 2 * precision * recall / (precision + recall + eps)

            # no-attack case: F1 = 1 only if all zeros predicted
            no_attack = (~t_nodes_b).all()
            perfect_zero = (pred_nodes.sum() == 0).float()
            temp_F1_SW = torch.where(no_attack, perfect_zero, f1)

            # strict match for attack samples; for no-attack, perfect_zero already used
            strict_match = (pred_nodes == t_nodes_b).all().long()
            localization_acc += torch.where(no_attack, (perfect_zero > 0).long(), strict_match)

            F1_95_SW += (temp_F1_SW >= 0.95).long()
            F1_5_SW  += (temp_F1_SW <= 0.05).long()

            # Stream per-bus
            TP_b += (pred_nodes &  t_nodes_b).long()
            FP_b += (pred_nodes & ~t_nodes_b).long()
            FN_b += (~pred_nodes &  t_nodes_b).long()

    # NW aggregation
    never_attacked = (TP_b == 0) & (FN_b == 0)
    F1_95_NW = (never_attacked & (FP_b == 0)).sum()
    F1_5_NW  = (never_attacked & (FP_b > 0)).sum()

    attacked = ~never_attacked
    if attacked.any():
        p_b = TP_b[attacked].float() / (TP_b[attacked] + FP_b[attacked] + eps)
        r_b = TP_b[attacked].float() / (TP_b[attacked] + FN_b[attacked] + eps)
        f1_b = 2 * p_b * r_b / (p_b + r_b + eps)

    F1_95_NW += (f1_b >= 0.95).long().sum()
    F1_5_NW  += (f1_b <= 0.05).long().sum()

    # Graph detection metrics
    det_FA     = gFP.float() / (gFP + gTN + eps)
    det_recall = gTP.float() / (gTP + gFN + eps)
    det_prec   = gTP.float() / (gTP + gFP + eps)
    det_F1     = 2 * det_prec * det_recall / (det_prec + det_recall + eps)

    return (
        det_recall.item(),
        det_FA.item(),
        det_F1.item(),
        (F1_95_SW.float() / n_batches).item(),
        (F1_5_SW.float()  / n_batches).item(),
        (F1_95_NW.float() / num_nodes).item(),
        (F1_5_NW.float()  / num_nodes).item(),
        (localization_acc.float() / n_batches).item(),
        (detection_acc.float()   / n_batches).item(),
        (total_loss / n_batches).item(),
    )

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