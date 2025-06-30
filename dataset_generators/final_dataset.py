import numpy as np
import torch
import torch.optim as optim
import pandapower as pp
import pandapower.networks as pn
import pandapower.topology as pptop
from pandapower.estimation import estimate
from pandapower.create import create_measurement
import networkx as nx
import copy
from scipy.optimize import minimize
from types import SimpleNamespace
from concurrent.futures import ProcessPoolExecutor, as_completed
import random
import os
import glob

# Implemented functions
from functions import saveNetwork, loadDataset, selectDevice

# TODO: to be deleteds
import sys

"""
Logic for non-sequential attacks:
    Every FDIA should not attack the next consequtive timestep to keep the attack more realistic
    and avoid potential overfitting to a seasonal dataset(By randomizing timestep, we avoid
                                                          seasonality)
    Save to the "final_dataset" folder is sequential.
"""



"""

TODOs:
    - Multiple random restarts of attack() if loss > TAU_LOSS
    - Lz might not affect the loss because measurement_model is not differentiable

    - Run the training without scaling
    - Run the training with the scaling
"""


dataset_config = {
    "normal_scaling": False,
    "max_timesteps": 44634,
    "number_nodes": 300
    }

# Create a uniformly shuffled list of numbers from a range 1 to max_timesteps
timesteps_left = list(range(1, dataset_config["max_timesteps"]))
random.shuffle(timesteps_left)

def returnTimeStep():
    return timesteps_left.pop()
    
    





# Hyperparameters for Ao
SIGMA_N = 0.005       # initial disturbance variance
LAMBDA_Z = 1.0        # weight for measurement loss
LAMBDA_Z = 1.0        # weight for state loss
ETA = 0.001           # learning rate for optimizer
EPOCHS = 1000         # number of attack optimization epochs
TAU_FREQ = 1.0        # attack frequency threshold
TAU_LOSS = 0.1        # attack quality (loss) threshold
R_MIN = 6             # minimum attack radius for IEEE300
R_MAZ = 8             # maximum attack radius for IEEE300
device = selectDevice()

# Utility: map complex state to pandapower net measurements
def measurement_model(x_complex, net):
    # Copy the net
    net_copy = copy.deepcopy(net)
    
    # Calculate voltage and angle for each bus
    mags = torch.abs(x_complex).detach().cpu().numpy()
    angs = torch.angle(x_complex).detach().cpu().numpy() * 180/np.pi
    
    # Assign voltage and angle to each bus
    net_copy.bus["vm_pu"] = mags
    net_copy.bus["va_degree"] = angs
    
    # Run power flow to get measurements at this fake operating point
    pp.runpp(net_copy, init='flat')
    
    # Extract bus measurements: p_mw, q_mvar, vm_pu, va_degree
    res = net_copy.res_bus[['p_mw','q_mvar','vm_pu','va_degree']].values.flatten()
    
    return res

# Build attack surface via BFS on IEEE300 graph
def bfs_attack_surface(net, entry_bus, radius):
    # Create a graph and generate BFS based on it
    G = pptop.create_nxgraph(net, respect_switches=False, include_impedances=False)
    lengths = nx.single_source_shortest_path_length(G, entry_bus, cutoff=radius)
    attacked_buses = list(lengths.keys())
    
    # Consider all buses
    all_buses = set(net.bus.index)
    
    # Find generator buses
    #    include all synchronous gens, static gens, and the external grid (slack)
    gen_buses = set()
    if hasattr(net, "gen"):
        gen_buses |= set(net.gen.bus.values)
    if hasattr(net, "sgen"):
        gen_buses |= set(net.sgen.bus.values)
    if hasattr(net, "ext_grid"):
        gen_buses |= set(net.ext_grid.bus.values)
    
    # Find buses with some injections
    inj_buses = set()
    for attr in ("load", "shunt", "gen", "sgen", "ext_grid"):
        if hasattr(net, attr):
            inj_buses |= set(getattr(net, attr).bus.values)
    
    # Find zero-injection buses: complement of buses with some injections
    zero_inj_buses = all_buses - inj_buses
    
    # Create a mask where 0-injection and generator buses are disregarded
    remove_buses = gen_buses | zero_inj_buses
    attacked_buses = [b for b in attacked_buses if b not in remove_buses]
    
    # State indices (one per bus)
    T_x = attacked_buses
    
    # Measurement indices: assume 4 measurements per bus
    T_z = []
    for b in attacked_buses:
        T_z.extend([4*b + i for i in range(4)])
    
    # Asserts to make sure T_x and T_z are aligned with the output  
    n = net.bus.shape[0]
    assert all(0 <= i < n    for i in T_x), "Tₓ out of [0,n)"
    assert all(0 <= j < 4*n  for j in T_z), "T_z out of [0,4n)"
    
    return T_z, T_x


def run_psse(net, z, sigma_n, nodelist):
    # Clear old measurements
    net.measurement = net.measurement.iloc[0:0]

    # Load measurements
    for idx, bus in enumerate(nodelist):
        p, q, vm, va = z[idx]
        create_measurement(net, "p", "bus", p, std_dev=abs(p)*sigma_n, element=bus)
        create_measurement(net, "q", "bus", q, std_dev=abs(q)*sigma_n, element=bus)
        create_measurement(net, "v", "bus", vm, std_dev=abs(vm)*sigma_n, element=bus)
        create_measurement(net, "va", "bus", va, std_dev=abs(va)*sigma_n, element=bus)

    # Run WLS estimation
    try:
        estimate(
            net,
            algorithm="wls",
            init="flat",
            tolerance=1e-6,
            maximum_iterations=20,
            calculate_voltage_angles=True
        )
    except Exception as e:
        print(f"[ERROR!]: The estimation didn't converge! {e}")

    # Build estimated feature matrix    
    cols = ["p_mw", "q_mvar", "vm_pu", "va_degree"]
    # reindex the estimator result to exactly nodelist (will insert NaNs for missing buses)
    est = net.res_bus_est.reindex(nodelist)
    # reindex the full power‐flow result
    pf  = net.res_bus.reindex(nodelist)
    # fill any NaNs in est with the pf values
    filled = est.fillna(pf)
    # now every bus is present, in order
    x = filled[cols].values   # shape (len(nodelist), 4)

    return x


class Attacker_Ao:
    def __init__(self, net):
        self.net = net
        self.n = None
        self.z0 = None
        self.T_z = None
        self.T_x = None
        self.x_hat_phasor = None
    
    # Define zero‐order objective on full loss L = λz·Lz – λx·Lx
    def obj(self, vth):
        V  = vth[:self.n]
        th = vth[self.n:]
        
        x_c = V * np.exp(1j * th)
        za  = measurement_model(torch.tensor(x_c, dtype=torch.complex64), self.net)
        
        # stealth loss on honest measurements
        mask_z = np.ones_like(za, bool);  
        mask_z[self.T_z] = False
        Lz = np.linalg.norm((za - self.z0)[mask_z], 2)
        
        # impact loss on honest states
        mask_x = np.ones(self.n, bool);  
        mask_x[self.T_x] = False
        Lx = np.linalg.norm((x_c - self.x_hat_phasor)[mask_x], 1)
        
        return LAMBDA_Z * Lz - LAMBDA_Z * Lx

    def attack(self, z0, x_hat_phasor, V, th, T_z, T_x):
        # Number of nodes
        self.n = len(x_hat_phasor)
        
        # Save values for reference from obj function
        self.T_z = T_z
        self.T_x = T_x
        self.x_hat_phasor = x_hat_phasor
        self.z0 = z0
        
        # Add noise to both V and th
        noise = np.random.normal(0, SIGMA_N, size=V[self.T_x].shape)
        V[self.T_x]  += noise
        th[self.T_x] += noise
        
        # Turn honest variables into torch variables:
        z0_t     = torch.tensor(self.z0, dtype=torch.float32, device=device)
        x_hat_t  = torch.tensor(self.x_hat_phasor,
                        dtype=torch.complex64,
                        device=device,
                        requires_grad=False)
        
        # Turn V and th to torch tensors:
        V = torch.nn.Parameter(torch.from_numpy(V).float().to(device))
        th = torch.nn.Parameter(torch.from_numpy(th).float().to(device))
        
        # Define optimizer and loss
        optimizer = torch.optim.SGD([V, th], lr=ETA)
        loss = None

        # Minimize loss from L = LAMBDA_Z * Lz - LAMBDA_Z * Lx
        for epoch in range(EPOCHS):
            optimizer.zero_grad()
        
            # clamp into [0.9,1.1] and [−π,π]
            V.data.clamp_(0.9, 1.1)
            th.data.clamp_(-np.pi, np.pi)
        
            # build phasors and analytic measurements
            x_complex = V * torch.exp(1j * th)
            za = torch.from_numpy(measurement_model(x_complex, self.net)).float().to(device)
        
            # compute per-paper losses
            mask_z = torch.ones(4*self.n, dtype=bool); 
            mask_z[self.T_z] = False
            mask_x = torch.ones(self.n,   dtype=bool); 
            mask_x[self.T_x] = False
        
            Lz = torch.norm( (za[mask_z] - z0_t[mask_z]), p=2 )
            Lx = torch.sum( torch.abs(x_complex[mask_x] - x_hat_t[mask_x]) )
        
            L = LAMBDA_Z * Lz - LAMBDA_Z * Lx
            L.backward()
            optimizer.step()
            
            loss = L.item()
        
            if loss < TAU_LOSS:
                break
        
        # Compute complex state variable and transform it to numpy
        x_c_opt = V * torch.exp(1j * th)
        x_c_opt = x_c_opt.cpu().detach().numpy()
        
        # Get new Z
        za_opt = measurement_model(torch.tensor(x_c_opt, dtype=torch.complex64), self.net)
        return za_opt, loss

# Generate function per timestep
def step_Ao( attacker, z0, x_hat, net, t):    
    # Initialize V and th
    V = abs(x_hat[:, 2])
    th = x_hat[:,3] * np.pi/180
    
    # Compute x_phasor
    x_hat_phasor = V * np.exp(1j * th)
    
    # Info required for an attack
    p = np.random.randint(0, x_hat.shape[0])
    r = np.random.randint(R_MIN, R_MAZ+1)
    T_z, T_x = bfs_attack_surface(net, p, r)
    
    # Attack attempt
    z, loss = attacker.attack(z0, x_hat_phasor, V, th, T_z, T_x)
    # — build a length-n vector of 0/1 labels —
    y = np.zeros_like(x_hat_phasor, dtype=int)
    y[T_x] = 1
    
    """
    print(loss)
    if loss > TAU_LOSS:
        # TODO: implement attack rerun
        print(f"[WARNING]: Sample #{t} didn't go below TAU_LOSS by {loss - TAU_LOSS}")
    """
    
    """
    # Copy the net
    net_copy = copy.deepcopy(net)
    
    # Build a nodelist matching the order of your z vector
    # IEEE300: it's just [0,1,2,…,299]
    nodelist = list(range(len(x_hat_phasor)))

    # Reshape z_out to (n,4)
    z_mat = z.reshape((len(x_hat_phasor), 4))

    # call your existing helper
    x_est = run_psse(net_copy, z_mat, SIGMA_N, nodelist)
    
    # run_psse returns None if it didn’t converge
    if x_est is None:
        print("[WARNING]: PSSE didn't converge!")
    """
    x_est = None
        
    return z, x_est, y


def worker_Ao(t_save, t_load):
    # IEEE300 network
    net = pn.case300()
    attacker = Attacker_Ao(net)

    # Input holders
    z0 = np.load(f"../init_dataset/Z_{t_load}.npy")
    x_hat = np.load(f"../init_dataset/Z_{t_load}.npy")
    
    # Flatten z0 for future calculations
    z0 = z0.flatten()
    
    z_att, x_est, y = step_Ao(attacker, z0, x_hat, net, t_save)
    
    # Save current attacked sample
        # Exclude Voltage magnitude and angle
    z_att = z_att.reshape(300, 4)
    z_att = z_att[:, :2]
    np.save(f'../final_dataset/Z_{t_save}.npy', z_att)
        
    # Estimeated X
    #np.save(f'../final_dataset/X_{t_save}.npy', x_est)
    
        # Attach "attacked" graph-level flag to the target
    y = np.concatenate([y, [1]])
    np.save(f'../final_dataset/target_{t_save}.npy', y)
    
    return t_save  # for tracking progress


def generate_Ao(start_id, finish_id, max_workers=12):
    
    # Un-paralleled generation for debugging
    """
    for t_save in range(start_id, finish_id):
        print("Ao attack chosen")
        t_load = returnTimeStep()
        worker_Ao(t_save, t_load)
        print(f"Current cycle: #{t_save}")
     """
    
    # Prepare timesteps for the parallel execution
    # ( in order to avoid undefined behavior )
    jobs = []
    for t_save in range(start_id, finish_id):
        t_load = returnTimeStep()
        #print(len(timesteps_left), t_load)
        jobs.append((t_save, t_load))
    
    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        futures = {
            exe.submit(worker_Ao, t_save, t_load): t_save
            for (t_save, t_load) in jobs
        }
        for fut in as_completed(futures):
            print("Ao attack chosen")
            t_save = futures[fut]
            try:
                fut.result()
                print(f"Current cycle: #{t_save}")
            except Exception as e:
                print(f"[✗] Error on sample {t_save}: {e}")


def generate_Ad(t_save):
    # Load a node features and a mask with the buses to be attacked marked as True
    t_load = returnTimeStep()
    Z, mask = loadDataset(t_load, True, dataset_config["number_nodes"])
  
    # - Ad data subset -
    print("Ad attack chosen")
        
    # Extract all node features
    P = Z[:, 0]
    Q = Z[:, 1]
  
    # Don't follow outliners, so that the attack is not as easy to detect
    # Get mean and STD for 90% of P
    lo, hi = np.percentile(P, [5,95])
    P_trim = P[(P>=lo)&(P<=hi)]
    P_mu, P_std  = P_trim.mean(), P_trim.std()
    
    # Get mean and STD for 90% of Q
    lo, hi = np.percentile(Q, [5,95])
    Q_trim = Q[(Q>=lo)&(Q<=hi)]
    Q_mu, Q_std  = Q_trim.mean(), Q_trim.std()
    
    # Attack buses that have some load only (otherwise => easy to detect)
    mask_P = mask & (P != 0)
    mask_Q = mask & (Q != 0)
  
    # Apply the distribution-based attack on the marked buses
    Z[mask_P, 0] = np.random.normal(P_mu, P_std, size=mask_P.sum())
    Z[mask_Q, 1] = np.random.normal(Q_mu, Q_std, size=mask_Q.sum())
    
    # Exclude Voltage magnitude and angle
    Z = Z[:, :2]
  
    # Generate target(expected output) for multi-label supervised learning
    target = np.concatenate([mask.astype(int), [1]])
  
    # Save the modified files
    saveNetwork(Z, target, t_save)


def generate_As(t_save):
    # Load a node features and a mask with the buses to be attacked marked as True
    t_load = returnTimeStep()
    Z, mask = loadDataset(t_load, True, dataset_config["number_nodes"])
  
    # - As data subset -
    print("As attack chosen")
      
    # Apply the scale-based attack on the marked buses
    Z[mask, 0] = Z[mask, 0] * np.random.uniform(0.8, 1.2, size=mask.sum())
    Z[mask, 1] = Z[mask, 1] * np.random.uniform(0.8, 1.2, size=mask.sum())
    
    # Exclude Voltage magnitude and angle
    Z = Z[:, :2]
  
    # Generate target(expected output) for multi-label supervised learning
    target = np.concatenate([mask.astype(int), [1]])
  
    # Save the modified files
    saveNetwork(Z, target, t_save)


def generate_Ar(t_save):
    # Load a node features and a mask with the buses to be attacked marked as True
    t_load = returnTimeStep()
    Z, mask = loadDataset(t_load, True, dataset_config["number_nodes"])
    
    # Load data TAU steps before
    # [NOTE]: assumes that the current step is grate than tau
    tau = 4
    Z_tau, _ = loadDataset(t-tau, True, dataset_config["number_nodes"])
  
    # - Ar data subset -
    print("Ar attack chosen")
      
    # Apply the data replay based attack on the marked buses
    Z[mask, 0] = Z_tau[mask, 0]
    Z[mask, 1] = Z_tau[mask, 1]
    
    # Exclude Voltage magnitude and angle
    Z = Z[:, :2]
  
    # Generate target(expected output) for multi-label supervised learning
    target = np.concatenate([mask.astype(int), [1]])
  
    # Save the modified files
    saveNetwork(Z, target, t_save)


def generate_honest(t):
    # Load a node features and a mask filled with False values
    Z, mask = loadDataset(t, False, dataset_config["number_nodes"])
    
    # Exclude Voltage magnitude and angle
    Z = Z[:, :2]
  
    # Generate target(expected output) for multi-label supervised learning
    target = np.concatenate([mask.astype(int), [0]])
  
    # Save the modified files
    saveNetwork(Z, target, t)




# # # # # # # # # # # #
# ---- Main code ---- #
# # # # # # # # # # # #

if __name__ == '__main__':
    # ! ---- Training set ---- !
    # -- Generate Ao --
    generate_Ao(1, 5761)   
    
    # -- Generate Ad --
    for t in range(5761, 11521):
      print(f"\n: Current cycle: #{t}")
      generate_Ad(t)
    
    # -- Generate honest samples (not-attacked) --
    for t in range(11521, 23041):
      print(f"\nNot-attacked, Current cycle: #{t}")
      generate_honest(t)
    
    # ! ---- Validation set ---- !
    # -- Generate Ao --
    generate_Ao(23041, 24481)   
    
    # -- Generate Ad --
    for t in range(24481, 25921):
      print(f"\n: Current cycle: #{t}")
      generate_Ad(t)
    
    # -- Generate honest samples (not-attacked) --
    for t in range(25921, 28801):
      print(f"\nNot-attacked, Current cycle: #{t}")
      generate_honest(t)
    
    # ! ---- Test set ---- !
    # -- Generate Ao --
    generate_Ao(28801, 29521)
    
    # -- Generate Ad --
    for t in range(29521, 30241):
      print(f"\n: Current cycle: #{t}")
      generate_Ad(t)
    
    # -- Generate As -- 
    for t in range(30241, 30961):
      print(f"\nCurrent cycle: #{t}")
      generate_As(t)
    
    # -- Generate Ar -- 
    for t in range(30961, 31681):
      print("[NOTE]: assumes that t > 10")
      print(f"\nCurrent cycle: #{t}")
      generate_Ar(t)
    
    # -- Generate honest samples (not-attacked) --
    for t in range(31681, 34561):
      print(f"\nNot-attacked, Current cycle: #{t}")
      generate_honest(t)
    
    
    # ! ---- Standardization ---- !
    root = "../final_dataset"
    pattern = os.path.join(root, "Z_*.npy")
    files = glob.glob(pattern)

    # Compute mean & std
    sum_   = 0.0
    sumsq  = 0.0
    count  = 0

    for fn in files:
        arr = np.load(fn)
        sum_  += arr.sum()
        sumsq += (arr**2).sum()
        count += arr.size

    mean = sum_ / count
    var  = sumsq / count - mean**2
    std  = np.sqrt(var)

    print(f"Global mean = {mean:.5f},  std = {std:.5f}")

    # Standardize each file
    for fn in files:
        arr = np.load(fn)
        arr = (arr - mean) / std
        base = os.path.basename(fn)
        np.save(os.path.join(root, base), arr)
