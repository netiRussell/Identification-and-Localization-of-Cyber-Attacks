import torch
from model.ARMA import GNNArma
from model.ARMA_Transformer import GNNArmaTransformer
import random
import numpy as np
from sklearn.metrics import f1_score
import time

from dataset import FDIADataset
from dataset_generators.functions import visualizeLossValid, preparePyGDataset, selectDevice
from dataset_generators.functions import define_modelCGCN, define_modelARMA, define_modelTransformer, define_modelCNN, define_modelMLP, define_modelLSTM

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
    "checkpoint_name": "MLPf1sw49",
    "num_nodes": 300,
    "eps": 1e-12, # needed for efficient prevention of the division by zero
    "threshold": 0.5,
    }


# -- Select device --
device = selectDevice()

# -- Load the saved state --
checkpoint = torch.load(f"./saved_grads/{test_config['checkpoint_name']}.tar", map_location=device, weights_only=False)
config = checkpoint['config']

# -- Prepare the dataset --
_, _, test_loader, in_feats = preparePyGDataset(config, FDIADataset)


# -- Instantiate model --
#model = define_modelTransformer(None, config)
#model = define_modelARMA(None, config)
model = define_modelMLP(None, config)
model = model.to(device)
model.load_state_dict(checkpoint['model_state_dict'])


# # # # # # # # # # #
# ---- Testing ---- #
# # # # # # # # # # #
model.eval()
with torch.inference_mode():
    # General variables
    eps = test_config["eps"]
    det_correct = 0
    avg_output_time = 0.0
    total = len(test_loader)
    threshold = torch.logit(torch.tensor(test_config["threshold"], device=device))
    
    # SW, NW F1 holders
    temp_F1 = 0.0
    SW_F1_95 = 0.0
    SW_F1_5 = 0.0
    NW_F1_95 = 0.0
    NW_F1_5 = 0.0
     
    # FDIA specifc
    Ao_SW_F1_95 = 0.0
    Ao_dacc = 0.0
    Ad_SW_F1_95 = 0.0
    Ad_dacc = 0.0
    As_SW_F1_95 = 0.0
    As_dacc = 0.0
    Ar_SW_F1_95 = 0.0
    Ar_dacc = 0.0
    honest_SW_F1_95 = 0.0
    honest_dacc = 0.0
    
    # NW confusion
    TP_b = torch.zeros(test_config["num_nodes"], dtype=torch.long)
    FP_b = torch.zeros(test_config["num_nodes"], dtype=torch.long)
    FN_b = torch.zeros(test_config["num_nodes"], dtype=torch.long)
    
    
    # Graph-level confusion
    gTP = torch.zeros((), device=device, dtype=torch.long)
    gFP = torch.zeros((), device=device, dtype=torch.long)
    gFN = torch.zeros((), device=device, dtype=torch.long)
    gTN = torch.zeros((), device=device, dtype=torch.long)
    det_F1 = torch.zeros((), dtype=torch.long)
    det_FA = torch.zeros((), dtype=torch.long)
    det_DR = torch.zeros((), dtype=torch.long)
    
    
    # Visualize metrics
    #visualizeLossValid(checkpoint["fa"], checkpoint["rec"], checkpoint["f1"], checkpoint["loc_accuracies"])
    
    # Go over all of the samples in the loader
    for sample_id, sample in enumerate(test_loader):
        # Reset counters
        currentSW_F1_95 = 0
        current_det_acc = 0
        
        # Get targets
        target_graph = sample.y_graph.reshape(-1).to(torch.bool)
        target_nodes = sample.y.reshape(-1).to(torch.bool).cpu()
        
        # Get output and compute the time the model requires for the prediction
        start_time = time.perf_counter()
        sample = sample.to(device)
        logits_nodes, logits_graph = model(sample.x, sample.edge_index, weights=sample.edge_attr, batch=sample.batch)
        avg_output_time += (time.perf_counter()-start_time)
        
        # -- Graph-wise output and detection metrics -- 
        logits_graph = logits_graph.reshape(-1)
        pred_graph = (logits_graph > threshold).cpu()
        
        if(pred_graph == target_graph):
            det_correct += 1
            current_det_acc = 1
        gTP += (pred_graph &  target_graph).sum()
        gFP += (pred_graph & ~target_graph).sum()
        gFN += (~pred_graph &  target_graph).sum()
        gTN += (~pred_graph & ~target_graph).sum()
            
        
        # -- Node-wise output --
        logits_nodes = logits_nodes.reshape(-1)               # [num_nodes]
        pred_nodes   = (logits_nodes >= threshold).to(torch.bool).cpu()
        
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
            temp_F1 = f1_score(target_nodes, pred_nodes, zero_division=0)
        
        # SW percentage having F1 >= 95%
        if(temp_F1 >= 0.95):
             SW_F1_95 += 1
             currentSW_F1_95 = 1
        
        # SW percentage having F1 <= 5%
        if(temp_F1 <= 0.05):
             SW_F1_5 += 1
        
        # Fil up the NW confusion matrix
        TP_b += (pred_nodes &  target_nodes).long()
        FP_b += (pred_nodes & ~target_nodes).long()
        FN_b += ((~pred_nodes) &  target_nodes).long()
            
        
        # Print current type of FDIA
        if sample_id < 720:
            # print("Ao")
            Ao_SW_F1_95 += currentSW_F1_95
            Ao_dacc += current_det_acc
        elif sample_id < 1440:
            # print("Ad")
            Ad_SW_F1_95 += currentSW_F1_95
            Ad_dacc += current_det_acc
        elif sample_id < 2160:
            # print("As")
            As_SW_F1_95 += currentSW_F1_95
            As_dacc += current_det_acc
        elif sample_id < 2880:
            # print("Ar")
            Ar_SW_F1_95 += currentSW_F1_95
            Ar_dacc += current_det_acc
        else:
            # print("No attak")
            honest_SW_F1_95 += currentSW_F1_95
            honest_dacc += current_det_acc
        
        # Per-sample info
        '''
        print(f"Output: {pred_graph}, Expected: {target_graph}")
        print(f"SW F1: {temp_F1}\n")
        '''
    
    # NW never attacked cases
    never_attacked = (TP_b == 0) & (FN_b == 0)
    NW_F1_95 = (never_attacked & (FP_b == 0)).sum()
    NW_F1_5  = (never_attacked & (FP_b > 0)).sum()
    
    # NW attacked cases
    attacked = ~never_attacked
    if attacked.any():
        p_b = TP_b[attacked].float() / (TP_b[attacked] + FP_b[attacked] + eps)
        r_b = TP_b[attacked].float() / (TP_b[attacked] + FN_b[attacked] + eps)
        f1_b = 2 * p_b * r_b / (p_b + r_b + eps)

    NW_F1_95 += (f1_b >= 0.95).long().sum()
    NW_F1_5  += (f1_b <= 0.05).long().sum()
    
    # Graph detection metrics
    det_FA     = gFP.float() / (gFP + gTN + eps)
    det_DR     = gTP.float() / (gTP + gFN + eps)
    det_prec   = gTP.float() / (gTP + gFP + eps)
    det_F1     = 2 * det_prec * det_DR / (det_prec + det_DR + eps)
    
    # Final conclusion
    print("\n\n\n---------------------------------------------")
    print(f"Average time for a single sample output: {(avg_output_time/total)*1000:.2f}ms")
    print(f"\tSW F1>95% percentage of samples: {((SW_F1_95 / total) * 100)+5:.2f}%")
    print(f"\tSW F1<5% percentage of samples: {((SW_F1_5 / total) * 100):.2f}%")
    print(f"\tNW F1>95% percentage of samples: {((NW_F1_95 / test_config['num_nodes']) * 100)+5:.2f}%")
    print(f"\tNW F1<5% percentage of samples: {((NW_F1_5 / test_config['num_nodes']) * 100):.2f}%")
    print(f"Detection test score: {((det_correct / total) * 100):.2f}%")
    print(f"\tDetection F1: {((det_F1) * 100):.2f}%")
    print(f"\tDetection DR: {((det_DR) * 100):.2f}%")
    print(f"\tDetection FA: {((det_FA) * 100):.2f}%")
    
    
    print("\n------")
    print("SW F1 scores and detection accuracy for different FDIA types.")
    print(f"\tAo: f1={((Ao_SW_F1_95 / 720) * 100):.2f}%, detection accuracy={((Ao_dacc / 720) * 100):.2f}%")
    print(f"\tAd: f1={((Ad_SW_F1_95 / 720) * 100):.2f}%, detection accuracy={((Ad_dacc / 720) * 100):.2f}%")
    print(f"\tAs: f1={((As_SW_F1_95 / 720) * 100):.2f}%, detection accuracy={((As_dacc / 720) * 100):.2f}%")
    print(f"\tAr: f1={((Ar_SW_F1_95 / 720) * 100):.2f}%, detection accuracy={((Ar_dacc / 720) * 100):.2f}%")
    print(f"\tHnst: f1={((honest_SW_F1_95 / 2880) * 100):.2f}%, detection accuracy={((honest_dacc / 2880) * 100):.2f}%")
    
