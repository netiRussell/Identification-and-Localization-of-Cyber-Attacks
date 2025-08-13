import torch
from model.ARMA import GNNArma
from model.ARMA_Transformer import GNNArmaTransformer
import random
import numpy as np
from sklearn.metrics import f1_score

from dataset import FDIADataset
from dataset_generators.functions import visualizeLossValid, preparePyGDataset, selectDevice

# TODO: delete at the final stage
import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Enable reproducibility
torch.backends.cudnn.deterministic = True
random.seed(123)
torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)


test_config = {
    "checkpoint_name": "ARMAElite"
    }


# -- Select device --
device = selectDevice()

# -- Load the saved state --
checkpoint = torch.load(f"./saved_grads/{test_config['checkpoint_name']}.tar", map_location=device, weights_only=False)
config = checkpoint['config']

# -- Prepare the dataset --
_, _, test_loader, in_feats = preparePyGDataset(config, FDIADataset)


# -- Instantiate model --
# Define the model

model = GNNArma(
    in_channels=config["in_channels"], 
    num_ARMA_layers=config["arma_specific"]["num_ARMA_layers"],
    hidden_channels=config["arma_specific"]["hidden_channels"],
    num_stacks=config["arma_specific"]["num_stacks"],
    num_layers=config["arma_specific"]["num_layers"], 
    dropout=config["arma_specific"]["dropout"],
    num_nodes=config["num_nodes"]
)
"""
model = GNNArmaTransformer(
    in_channels=config["in_channels"],
    hidden_channels=config["hidden_channels"],
    out_channels=config["out_channels"],
    num_stacks=config["num_stacks"], 
    num_layers=config["num_layers"],
    transformer_heads=config["transformer_heads"],
    transformer_layers=config["transformer_layers"],
    dropout=config["dropout"],
    num_nodes=config["num_nodes"],
    leakyRelu_rate=config["leakyRelu_rate"]
)
"""

model = model.to(device)
model.load_state_dict(checkpoint['model_state_dict'])


# # # # # # # # # # #
# ---- Testing ---- #
# # # # # # # # # # #
model.eval()
with torch.no_grad():
    det_correct = 0
    F1_95 = 0.0
    temp_F1 = 0.0
    total = len(test_loader)
    
    Ao_F1_95 = 0.0
    Ao_dacc = 0.0
    Ad_F1_95 = 0.0
    Ad_dacc = 0.0
    As_F1_95 = 0.0
    As_dacc = 0.0
    Ar_F1_95 = 0.0
    Ar_dacc = 0.0
    honest_F1_95 = 0.0
    honest_dacc = 0.0
    
    threshold = 0.4
    
    
    # Visualize metrics
    visualizeLossValid(checkpoint["fa"], checkpoint["rec"], checkpoint["f1"], checkpoint["loc_accuracies"])
    
    # Go over all of the samples in the loader
    for sample_id, sample in enumerate(test_loader):
        # Reset counters
        currentF1_95 = 0
        current_det_acc = 0
        
        # Get targets
        target_graph = sample.y_graph.item()
        target_nodes = sample.y
        
        # Get output
        sample = sample.to(device)
        logits_nodes, logits_graph = model(sample.x, sample.edge_index, weights=sample.edge_attr, batch=sample.batch)
        
        # -- Graph-wise output -- 
        probs = torch.sigmoid(logits_graph)
        classification = 1 if probs[0] >= threshold else 0
        
        if(int(classification) == int(sample.y_graph.item())):
            det_correct += 1
            current_det_acc = 1
            
        
        # -- Node-wise output --
        pred_nodes = (torch.sigmoid(logits_nodes) >= threshold).float()
        
        if( target_graph == 0 ):
            # - No attack case -
            
            # If prediction is 100% correct (all nodes = 0 since no attack)
            if(pred_nodes.sum() == 0):
                temp_F1 = 1.0
                
            # If at least 1 node is a mismatch (has a value of 1)    
            else:
                temp_F1 = 0.0
        else:
            # - Attack took place case -
            # Compute F1
            temp_F1 = f1_score(target_nodes, pred_nodes.view(-1).cpu(), zero_division=0)
        
        # Sample percentage having F1 >= 95%
        if(temp_F1 >= 0.95):
             F1_95 += 1
             currentF1_95 = 1
            
        
        # Print current type of FDIA
        if sample_id < 720:
            print("Ao")
            Ao_F1_95 += currentF1_95
            Ao_dacc += current_det_acc
        elif sample_id < 1440:
            print("Ad")
            Ad_F1_95 += currentF1_95
            Ad_dacc += current_det_acc
        elif sample_id < 2160:
            print("As")
            As_F1_95 += currentF1_95
            As_dacc += current_det_acc
        elif sample_id < 2880:
            print("Ar")
            Ar_F1_95 += currentF1_95
            Ar_dacc += current_det_acc
        else:
            print("No attak")
            honest_F1_95 += currentF1_95
            honest_dacc += current_det_acc
        
        # Per-sample info
        print(f"Output: {probs.item()}({classification}), Expected: {target_graph}")
        print(f"Current F1: {temp_F1}\n")
    
    # Final conclusion
    print("---------------------------------------------")
    print(f"TOTAL({total}) F1>95% percentage of samples: {((F1_95 / total) * 100):.2f}%")
    print(f"TOTAL({total}) Detection test score: {((det_correct / total) * 100):.2f}%\n")
    
    print(f"Ao: f1={((Ao_F1_95 / 720) * 100):.2f}%, detection accuracy={((Ao_dacc / 720) * 100):.2f}%")
    print(f"Ad: f1={((Ad_F1_95 / 720) * 100):.2f}%, detection accuracy={((Ad_dacc / 720) * 100):.2f}%")
    print(f"As: f1={((As_F1_95 / 720) * 100):.2f}%, detection accuracy={((As_dacc / 720) * 100):.2f}%")
    print(f"Ar: f1={((Ar_F1_95 / 720) * 100):.2f}%, detection accuracy={((Ar_dacc / 720) * 100):.2f}%")
    print(f"Honest: f1={((honest_F1_95 / 2880) * 100):.2f}%, detection accuracy={((honest_dacc / 2880) * 100):.2f}%")
    
