"""
This code gathers Ao losses for the network for further analysis and generation of a threshold
"""

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
from collections import deque

# Implemented functions
from functions import saveNetwork, load_honest, load_tobeAttacked, selectDevice


# # # # # # # # # # # # # #  #
# ---- Global variables ---- #
# # # # # # # # # # # # # #  #

# Avoid scientific notation
torch.set_printoptions(sci_mode=False)

dataset_config = {
    "normal_scaling": False,
    "max_timesteps": 132476+1,
    "standardize_data": True,
    
    "number_nodes": 118,
    "net_template": pn.case118(), # ! change seperately in the worker_Ao function
    'init_dataset_path': "../init_datasetIEEE118",
    'final_dataset_path': "../final_datasetIEEE118"
    }

dataset_config["G"] = pptop.create_nxgraph(dataset_config["net_template"], respect_switches=False, include_impedances=False)

# Deque for loss
loss_deque = deque()

# Create a uniformly shuffled list of numbers from a range 1 to max_timesteps
timesteps_left = deque([i for i in range(1, dataset_config["max_timesteps"])])
#timesteps_left = list(range(1, dataset_config["max_timesteps"]))
random.shuffle(timesteps_left)

def returnTimeStep():
    if( len(timesteps_left) == 0 ):
        raise ValueError("ALL THE TIMESTEPS HAVE BEEN USED")
    
    return timesteps_left.pop()

def appendTimeStepToTheEnd(t_load):
    timesteps_left.appendleft(t_load)
    
# -- Create a mask of non-attackable buses --
# Consider all buses
all_buses = set(dataset_config["net_template"].bus.index)

# Find generator buses
#    include all synchronous gens, static gens, and the external grid (slack)
gen_buses = set()
if hasattr(dataset_config["net_template"], "gen"):
    gen_buses |= set(dataset_config["net_template"].gen.bus.values)
if hasattr(dataset_config["net_template"], "sgen"):
    gen_buses |= set(dataset_config["net_template"].sgen.bus.values)
if hasattr(dataset_config["net_template"], "ext_grid"):
    gen_buses |= set(dataset_config["net_template"].ext_grid.bus.values)

# Find buses with some injections
inj_buses = set()
for attr in ("load", "shunt", "gen", "sgen", "ext_grid"):
    if hasattr(dataset_config["net_template"], attr):
        inj_buses |= set(getattr(dataset_config["net_template"], attr).bus.values)

# Find zero-injection buses: complement of buses with some injections
zero_inj_buses = all_buses - inj_buses

# Create a mask where 0-injection and generator buses are included
irrelevant_buses = gen_buses | zero_inj_buses


# Hyperparameters for Ao
SIGMA_N = 0.005       # initial disturbance variance
LAMBDA_Z = 1.0        # weight for measurement loss
LAMBDA_X = 1.0        # weight for state loss
ETA = 0.001           # learning rate for optimizer
EPOCHS = 1000         # number of attack optimization epochs
TAU_FREQ = 1.0        # attack frequency threshold
TAU_LOSS = 0.1        # attack quality (loss) threshold
R_MIN = 3             # minimum attack radius for 2:IEEE14, 3:IEEE118, 6:IEEE300
R_MAX = 4             # maximum attack radius for 3:IEEE14, 4:IEEE118, 8:IEEE300
SENSOR_SIGMA = 0.01   # match the sigma used in the init_dataset
TAU_OUTLINER = 3.7      # minimal loss to be considered a well generated sample (1, for IEEE14, 3.7 for IEEE118, 8 for IEEE300)
device = selectDevice()


# # # # # # # # # # # #
# ---- Functions ---- #
# # # # # # # # # # # #

# Utility: map complex state to net measurements
def measurement_model(x_complex, Ybus, baseMVA):
    # Compute complex power injection S = V * conj(I) = V * conj(Ybus @ V)
    S = x_complex * torch.conj(Ybus @ x_complex)
    
    # Extract real and imaginary parts
    P = -S.real.float()*baseMVA      # active‐power injections
    Q = -S.imag.float()*baseMVA      # reactive‐power injections
    
    # Direct measurements
    V_mag = torch.abs(x_complex).float()
    angle_rad = torch.angle(x_complex).float()
    
    # Convert angle to degrees
    angle_deg = angle_rad * (180.0 / torch.pi)
    
    # Extract bus measurements: p_mw, q_mvar, vm_pu, va_degree
    return torch.stack([P, Q, V_mag, angle_deg], dim=1)  # (n,4)

# Build attack surface via BFS on IEEE300 graph
def bfs_attack_surface(net, entry_bus, radius):
    
    _, _, T_x = load_tobeAttacked(dataset_config["G"], irrelevant_buses, 1, entry_bus, radius,
                                  dataset_config["number_nodes"], dataset_config["init_dataset_path"])
    
    if( T_x is None ):
        return None, None
    
    # Measurement indices: assume 4 measurements per bus
    T_z = []
    for b in T_x:
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
        return None

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
    def __init__(self, Ybus, baseMVA, mu_pq, cov_inv):
        self.Ybus = Ybus
        self.baseMVA = baseMVA
        self.mu_pq = torch.as_tensor(mu_pq, dtype=torch.float32, device=device) 
        self.cov_inv = torch.as_tensor(cov_inv, dtype=torch.float32, device=device)
    

    def attack(self, z0, x_hat_phasor, V, th, T_z, T_x):
        # Number of nodes
        n = len(x_hat_phasor)
        
        # Add noise to both V and th
        noise = np.random.normal(0, SIGMA_N, size=V[T_x].shape)
        V[T_x]  += noise
        th[T_x] += noise
        
        # Turn honest variables into torch variables:
        z0_t     = torch.tensor(z0, dtype=torch.float32, device=device)
        x_hat_t  = torch.tensor(x_hat_phasor,
                        dtype=torch.complex64,
                        device=device,
                        requires_grad=False)
        
        # Turn V and th to torch tensors:
        V = torch.nn.Parameter(torch.from_numpy(V).float().to(device))
        th = torch.nn.Parameter(torch.from_numpy(th).float().to(device))
        
        # Define optimizer and loss
        optimizer = torch.optim.Adam([V, th], lr=ETA)
        loss = None
        
        # Define masks for T_z and T_x
        attacked = torch.zeros(n, dtype=torch.bool, device=device)
        attacked[T_x] = True
        keep = ~attacked  # non-attacked buses
        
        # Stealth hyperparams
        EPS_KEEP   = 1e-4        # RMSE threshold on non-attacked
        BUDGET_REL = 0.03        # 3% cap on |Δ(P,Q)|_∞ at attacked
        CHI2_CUT   = 11.83       # ~99.7% for df=2 (Mahalanobis gate)

        # Minimize loss from L = LAMBDA_Z * Lz - LAMBDA_Z * Lx
        for epoch in range(EPOCHS):
            # The goal of this training is to find a new x̃ that:
            #   Maximally corrupts the measurements in Tₓ (the attacked nodes),
            #   While keeping the measurements outside Tₓ (i.e. all i∉Tₓ) as close as possible to the originals.
            
            optimizer.zero_grad()
        
            # Clamp into [0.9,1.1] and [−π,π]
            V.data.clamp_(0.9, 1.1)
            th.data.clamp_(-np.pi, np.pi)
        
            # Build phasors and analytic measurements
            x_complex = V * torch.exp(1j * th)
            za = measurement_model(x_complex, self.Ybus, self.baseMVA)
            z0_mat = z0_t.view(n, 4)   # reshape honest z
            
            # Lz: RMSE on NON-ATTACKED sensors
            e_keep = za[keep] - z0_mat[keep]
            Lz = torch.sqrt(torch.mean(e_keep**2) + 1e-12) 
            
            # Lx: MAE on ATTACKED states
            Lx = torch.mean(torch.abs(x_complex[attacked] - x_hat_t[attacked]))
            
            # Total loss
            L = LAMBDA_Z * Lz - LAMBDA_X * Lx
            L.backward()
            
            # zero out grads in the non-attacked nodes
            V.grad  *= attacked.float()
            th.grad *= attacked.float()
            
            # Update parameters based on the gradients (only nodes from T_x == 1 have influence)
            optimizer.step()
            
            loss = L.item()
            
            # Early-stop when stealth criteria already met (does not change objective)
            with torch.no_grad():
                Zpq  = za[:, :2]
                Z0pq = z0_mat[:, :2]
            
                eps_rel = 1e-6
                dz_rel = torch.abs((Zpq - Z0pq)[attacked]) / (torch.abs(Z0pq[attacked]) + eps_rel)
                budget_ok = (dz_rel.max().item() <= BUDGET_REL)
            
                rmse_keep_ok = (Lz.item() < EPS_KEEP)
            
                maha_ok = True
                if (self.mu_pq is not None) and (self.cov_inv is not None):
                    diff = (Zpq[attacked] - self.mu_pq[attacked])
                    d2   = torch.einsum('bi,bij,bj->b', diff, self.cov_inv[attacked], diff)
                    CHI2_CUT = 11.83
                    maha_ok = bool((d2 <= CHI2_CUT).all())
            
                if rmse_keep_ok and budget_ok and maha_ok:
                    break
        
        # Save the loss
        loss_deque.append(loss)
        
        return None, None

# Generate function per timestep
def step_Ao( attacker, z0, x_hat, net, t):    
    # Initialize V and th
    V = abs(x_hat[:, 2])
    th = x_hat[:, 3] * np.pi/180
    
    # Compute x_phasor
    x_hat_phasor = V * np.exp(1j * th)
    
    # Info required for an attack
    p = np.random.randint(0, x_hat.shape[0])
    r = np.random.randint(R_MIN, R_MAX+1)
    T_z, T_x = bfs_attack_surface(net, p, r)
    if( T_z is None and T_x is None):
        return None, None, None
    
    # Attack attempt
    z, loss = attacker.attack(z0, x_hat_phasor, V, th, T_z, T_x)
    if( loss is None ):
        return None, None, None
    
    # — build a length-n vector of 0/1 labels —
    y = np.zeros_like(x_hat_phasor, dtype=int)
    y[T_x] = 1
    
    # Add 1% relative uniform noise to match the other FDIAs 
    # (init_dataset adds noise to Z and other FDIAs reuse Z but Ao doesn't) 
    z = np.random.normal(loc=z, scale=abs(z * SENSOR_SIGMA))
    
    # Build a nodelist matching the order of your z vector
    # IEEE300: it's just [0,1,2,…,299]
    nodelist = list(range(len(x_hat_phasor)))

    # Reshape z_out to (n,4)
    z_mat = z.reshape((len(x_hat_phasor), 4))

    # call your existing helper
    x_est = run_psse(net, z_mat, SENSOR_SIGMA, nodelist)
    
    # run_psse returns None if it didn’t converge
    if( x_est is None ):
        return None, None, None
    
    # Exclude Voltage magnitude and angle from the measruments
    z = z[:, :2]
    
    # Exclude Voltage magnitude and angle from the estimated state
    x_est = x_est[:, :2]
        
    return z, x_est, y


def worker_Ao(t_save, t_load, mu, cov_inv):
    # Generate Ybus and base unit of measurements from the current network
    net = pn.case118()
    pp.runpp(net)                              # just to build net._ppc
    
    Y_ppc = net._ppc['internal']['Ybus'].toarray()
    pd2ppc = net._pd2ppc_lookups['bus']        # map: pandapower index -> ppc index
    Y_aligned = Y_ppc[pd2ppc][:, pd2ppc]       # reindex Ybus into pandapower order
    
    Ybus = torch.tensor(Y_aligned, dtype=torch.complex64, device=device)
    baseMVA = net.sn_mva

    # Initialize the attacker with the Ybus and the unit computed
    attacker = Attacker_Ao(Ybus, baseMVA, mu, cov_inv)

    # Input holders
    z0 = np.load(f"{dataset_config['init_dataset_path']}/Z_{t_load}.npy")
    x_hat = np.load(f"{dataset_config['init_dataset_path']}/X_{t_load}.npy")
    
    # Flatten z0 for future calculations
    z0 = z0.flatten()
    
    # Compute Z and X
    z_att, x_est, y = step_Ao(attacker, z0, x_hat, net, t_save)
    if( y is None ):
        return None
    
    
    # Save current attacked sample
    np.save(f'{dataset_config["final_dataset_path"]}/Z_{t_save}.npy', z_att)
        
    # Estimated X
    np.save(f'{dataset_config["final_dataset_path"]}/X_{t_save}.npy', x_est)
    
    # Attach "attacked" graph-level flag to the target
    y = np.concatenate([y, [1]])
    np.save(f'{dataset_config["final_dataset_path"]}/target_{t_save}.npy', y)
    
    return t_save  # for tracking progress


def generate_Ao(start_id, finish_id, mu, cov_inv, max_workers=2):
    
    # Un-paralleled generation for debugging
    # [WARNING] the logic is prone to using up all of the timesteps if TAU_OUTLINER is too low
    for t_save in range(start_id, finish_id):
        print("Ao attack chosen")
        
        # First attempt
        t_load = returnTimeStep()
        worker_Ao(t_save, t_load, mu, cov_inv)
        
        print(f"Current cycle: #{t_save}, Timesteps left: {len(timesteps_left)}")
 

def generate_honest(t):
    # Get a random timestep
    t_load = returnTimeStep()
    
    # Load a node features and a mask filled with False values
    Z, mask = load_honest(t_load, dataset_config["number_nodes"], dataset_config["init_dataset_path"])
    
    # Exclude Voltage magnitude and angle
    Z = Z[:, :2]
  
    # Generate target(expected output) for multi-label supervised learning
    target = np.concatenate([mask.astype(int), [0]])
  
    # Save the modified files
    saveNetwork(Z, target, t, dataset_config["final_dataset_path"])




# # # # # # # # # # # #
# ---- Main code ---- #
# # # # # # # # # # # #

if __name__ == '__main__':
    import multiprocessing as mp
    
    mp.set_start_method('spawn', force=True)
    
    # ! ---- Training set ---- !
    # -- Generate honest samples (not-attacked) --
    for t in range(1, 40000):
      print(f"\nNot-attacked, Current cycle: #{t}")
      generate_honest(t)  
      
    # -- Get per-bus mean and covariance for (P,Q) --
    # Collect healthy files
    Z_files = []
    for i in range(1, 40000):
        fn = f"{dataset_config['init_dataset_path']}/Z_{i}.npy"
        if not os.path.exists(fn):
            raise RuntimeError(f"{fn} missing")
        Z_files.append(fn)
    
    # Initialize using the first file to get n (buses)
    A0 = np.load(Z_files[0]).astype(np.float64)[:, :2] # shape (n, 2) -> (P,Q) per bus
    n = A0.shape[0]
    
    mu_sum = np.zeros((n, 2), dtype=np.float64)      # Σ x
    xx_sum = np.zeros((n, 2, 2), dtype=np.float64)   # Σ (x x^T)
    m = 0                                            # number of timesteps
    
    for fn in Z_files:
        A = np.load(fn).astype(np.float64)[:, :2]            # (n,2)
        mu_sum += A
        xx_sum += A[:, :, None] * A[:, None, :]      # outer product per bus -> (n,2,2)
        m += 1
    
    # Per-bus mean
    mu = mu_sum / m                                  # (n,2)
    
    # Per-bus covariance
    # sample version
    cov_sam = (xx_sum - m * (mu[:, :, None] * mu[:, None, :])) / (m - 1)
    
    # Stable inverse for Mahalanobis via eigen-floor
    eps = 1e-6
    cov_inv = np.empty_like(cov_sam)
    for j in range(n):
        w, V = np.linalg.eigh(cov_sam[j])
        w = np.clip(w, eps, None)                   # floor tiny/neg eigenvalues
        cov_inv[j] = (V * (1.0 / w)) @ V.T
    
    print("\nPer-bus covarience and mean have been computed. Starting the Ao generation...")
    
    # -- Generate Ao (Train) --    
    generate_Ao(40000, 132476, mu, cov_inv) 
    
    # Convert to numpy array
    arr = np.array(list(loss_deque), dtype=float)
    
    # Save
    np.save("loss_deque.npy", loss_deque)
    
   