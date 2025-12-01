import numpy as np
import glob
from collections import deque
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from t_sne import tsne
import shutil
import os
import sys # TODO: delete after debugging

# init dataset works on yields z to be utilized for the final dataset generation
# final dataset yields x to be utilized in the training

# TODO: add a t-sne check before and after

config = {
        # Paths to the datasets where X_*.npy files are stored
        "dataset_root_healthy": "./init_datasetIEEE118",
        "dataset_root_attacked": "./final_datasetIEEE118",
        
        # 1 = init dataset, find outliers among z
        # 2 = final dataset, find outliers among x
        "mode": 1,
        
        # Should the found outliers be moved?
        "move_outliers": True,
        
        # Show t-SNE pre and post
        "show_tsne": False,
        
        # p. u. limits
        "vm_min": 0.9,
        "vm_max": 1.1,
        
        # deg, abs angle diff limit
        "dtheta_max": 30,
        
        # robust power thresholds (quantiles)
        "p_low_q": 0.001,
        "p_high_q": 0.999,
        "q_low_q": 0.001,
        "q_high_q": 0.999,
        
        # IF hyperparams
        "IF_contamination": 0.01,
        "IF_n_estimators": 200,
        "IF_random_state": 123,
        "IF_max_train_samples": 20000, # subsample size for IF training (None = all)
        "IF_batch_size": 512, # number of samples to load in per IF predicting cycle
    }


def get_pq_bounds(files, num_nodes):
    all_p = np.zeros((len(files), num_nodes))
    all_q = np.zeros((len(files), num_nodes))
    for i, fp in enumerate(files):
        X = np.load(fp)      # (num_nodes, 4)
        p = X[:, 0]
        q = X[:, 1]
        all_p[i] = p
        all_q[i] = q
    
    all_p = np.concatenate(all_p, axis=0)
    all_q = np.concatenate(all_q, axis=0)
    
    p_low, p_high = np.quantile(all_p, [config["p_low_q"], config["p_high_q"]])
    q_low, q_high = np.quantile(all_q, [config["q_low_q"], config["q_high_q"]])
    
    print("p bounds:", p_low, p_high)
    print("q bounds:", q_low, q_high)
    
    return p_low, p_high, q_low, q_high

# Physics check
def physics_checks(X, adjacency, p_low, p_high, q_low, q_high):
    """
    X: (num_nodes, 4) with order:
       0: p_mw, 1: q_mvar, 2: vm_pu, 3: va_degree
    adjacency: (num_nodes, num_nodes) binary (0/1), symmetric, 1 = line present
    returns: boolean that represents whether the sample is invalid
    """

    num_nodes, num_feat = X.shape
    assert num_feat == 4

    p = X[:, 0]      # MW
    q = X[:, 1]      # Mvar
    vm = X[:, 2]     # p.u.
    va = X[:, 3]     # deg

    # Voltage magnitude range
    vm_bad = ((vm < config["vm_min"]) | (vm > config["vm_max"])).any()

    # Checking p and q using quantiles
    if (p_low is not None) and (p_high is not None):
       p_bad = ((p < p_low) | (p > p_high)).any()
    else:
       p_bad = False

    if (q_low is not None) and (q_high is not None):
       q_bad = ((q < q_low) | (q > q_high)).any()
    else:
       q_bad = False

    # Angle difference check,
    adjacency = np.asarray(adjacency)
    # indices of edges (i, j) with i < j
    i_idx, j_idx = np.where(np.triu(adjacency, k=1) > 0)
    if i_idx.size == 0:
        print("[WARNING]: adjacency has no edges > skipping angle-diff check")
        dtheta_bad = False
    else:
        va_i = va[i_idx]
        va_j = va[j_idx]
        dtheta = np.abs(va_i - va_j)
        dtheta_bad_samples = (dtheta > config["dtheta_max"]).any()
        dtheta_bad = dtheta_bad_samples

    return vm_bad | p_bad | q_bad | dtheta_bad

# Isolation forest


# Main code
if __name__ == "__main__":
    
    # Collect file list
    if( config['mode'] == 1 ):
        files = sorted(glob.glob(f"{config['dataset_root_healthy']}/X_*.npy"))
        root = config['dataset_root_healthy']
    elif( config['mode'] == 2 ):
        files = sorted(glob.glob(f"{config['dataset_root_attacked']}/X_*.npy"))
        root = config['dataset_root_attacked']
    else:
        raise RuntimeError("Invalid mode provided in the config{}")
        
    if not files:
        raise RuntimeError(f"No X_*.npy files in the {root}")
    num_samples = len(files)
    print("Total samples:", num_samples)
    
    # Outlier treatment for the healthy dataset
    if( config['mode'] == 1 ):
        # Generate the adjacency matrix from weights and edges
        edges = np.load(f"{config['dataset_root_healthy']}/edge_indices.npy")
        
        num_nodes = int(edges.max()) + 1
    
        adjacency = np.zeros((num_nodes, num_nodes), dtype=int)
        
        for k in range(edges.shape[1]):
            i = edges[0, k]
            j = edges[1, k]
            adjacency[i, j] = 1
            adjacency[j, i] = 1  # undirected grid
        print("Adjacency matrix has been generated.")
        
        # Initial t-SNE
        if( config['show_tsne'] ):
            tsne(files, num_nodes, "t-SNE Projection (pre)")
    
        # Get the p and q bounds.
        # Done separately to improve the memory efficiency on large datasets
        print("Cacluating P and Q bounds...")
        p_low, p_high, q_low, q_high = get_pq_bounds(files, num_nodes)
    
        # Go over each sample and apply the physics checks
        print("Physics check started")
        valid_IDs = deque()
        for i in range(1, num_samples+1):
            result = physics_checks(np.load(f"{config['dataset_root_healthy']}/X_{i}.npy"), 
                                    adjacency, p_low, p_high, q_low, q_high)
            
            # In case the sample is valid, save its ID
            if(~result):
                valid_IDs.append(i)
        
        # Build matrix for IsolationForest only on physics-valid samples
        
        # Convert to numpy array
        valid_IDs = np.array(valid_IDs)
        print(f"{len(valid_IDs)} out of {num_samples} samples have passed the physics checks")
        
        # Randomize
        rng = np.random.default_rng(0)
    
        # Get IDs for the IF training subset 
        print("Preparing the data for the IF alg.")
        if len(valid_IDs) > config["IF_max_train_samples"]:
            train_IDs = rng.choice(valid_IDs, size=config["IF_max_train_samples"], replace=False)
        else:
            train_IDs = valid_IDs.copy()
            
        # Build the IF training subset
        X_train_flat = np.zeros((len(train_IDs), num_nodes*4))
        for i, idx in enumerate(train_IDs):
            X = np.load(f"{root}/X_{idx}.npy")
            X_train_flat[i, :] = X.reshape(-1) # (num_nodes*4,)

        # Standardize
        scaler = StandardScaler()
        X_train_scaled  = scaler.fit_transform(X_train_flat)
        
        # Train the Isolation Forest algorithm
        print("IF training has been started")
        IF = IsolationForest(
            n_estimators=config["IF_n_estimators"],
            contamination=config["IF_contamination"],
            random_state=config["IF_random_state"],
        )
        IF.fit(X_train_scaled)
        
        # Apply the Isolation Forest algorithm in batches
        print("IF prediction has been started")
        batch_size = config["IF_batch_size"]
        scores = np.zeros(len(valid_IDs))  # anomaly scores for all physics-valid samples
        labels = np.zeros(len(valid_IDs), dtype=int)
        
        for start in range(0, len(valid_IDs), batch_size):
            end = start + batch_size
            batch_ids = valid_IDs[start:end]
        
            X_batch_flat = []
            for idx in batch_ids:
                X = np.load(os.path.join(root, f"X_{idx}.npy"))
                X_batch_flat.append(X.reshape(-1))
        
            X_batch_flat = np.stack(X_batch_flat, axis=0)
            X_batch_scaled = scaler.transform(X_batch_flat)
        
            scores_raw = IF.decision_function(X_batch_scaled)  # higher = more normal
            preds = IF.predict(X_batch_scaled)                 # 1 = inlier, -1 = outlier
        
            scores[start:end] = -scores_raw                     # higher = more anomalous
            labels[start:end] = preds

        print("Done with the outliers detection")
    
        # Results analysis
        print(f"Number of inliers: {np.sum(labels == 1)}")
        
        # Build inlier file list (physics-valid AND IF inliers)
        inlier_ids = valid_IDs[labels == 1]  # IDs that passed physics + IF
        inlier_files = [os.path.join(root, f"X_{idx}.npy") for idx in inlier_ids]
        
        # Final t-SNE
        if( config['show_tsne'] ):
            tsne(inlier_files, num_nodes, "t-SNE Projection (post)")
        
        # Delete the outliers
        if(config["move_outliers"]):
            outlier_dir = os.path.join(root, "outliers")
            os.makedirs(outlier_dir, exist_ok=True)
            
            # outliers: IF predicted -1 and physics invalid
            outlier_indices = np.array(list( set(range(1, num_samples+1)) - set(inlier_ids) ))
            
            for idx in outlier_indices:
                src = os.path.join(root, f"X_{idx}.npy")
                dst = os.path.join(outlier_dir, f"X_{idx}.npy")
                shutil.move(src, dst)
            
            print(f"{len(outlier_indices)} outlier samples moved to {outlier_dir}")