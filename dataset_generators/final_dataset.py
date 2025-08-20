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
from functions import saveNetwork, load_honest, load_tobeAttacked, selectDevice

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
    - Stats without standardization:
    - Stats with standardization:
    - Stats with cyclic lr:
    
    - Try not to standardize(copy the unstandardized data aside, train on the standardized one and then on the regular)
    - I think generators are skipped to be attacked but are still marked as attacked in the generated targets
        Check on Ao
        Check on regular attacks
    
    - Make sure that target matches the attacked nodes
    - Confirm Ao is implemented correctly. To do so, go over the notes in attacker and the paper
    
    - (warm‐up + cosine decay) for scheduler
    - Figure out why the gradients blow up so much
    - No layer norm for ARMA?
    
    
    - Multiple random restarts of attack() if loss > TAU_LOSS
    - Lz might not affect the loss because measurement_model is not differentiable
    
    - Make sure that there is no NaN in ARMATransformer. Perhaps this is one
        of the reasons why my models don't fully converge
        I added to the ARMA tranformer model:
            if torch.isnan(x).any() or torch.isnan(weights).any():
            print("Found NaN in inputs")
        I added to the Arma transformer training:
            torch.autograd.set_detect_anomaly(True)
    - Figure out the best lr for ARMA with optuna
"""

# # # # # # # # # # # # # #  #
# ---- Global variables ---- #
# # # # # # # # # # # # # #  #

# Avoid scientific notation
torch.set_printoptions(sci_mode=False)

dataset_config = {
    "normal_scaling": False,
    "max_timesteps": 44634+1,
    "number_nodes": 300,
    "net_template": pn.case300(),
    "standardize_data": True,
    }

dataset_config["G"] = pptop.create_nxgraph(dataset_config["net_template"], respect_switches=False, include_impedances=False)

# Create a uniformly shuffled list of numbers from a range 1 to max_timesteps
timesteps_left = list(range(1, dataset_config["max_timesteps"]))
random.shuffle(timesteps_left)

def returnTimeStep():
    return timesteps_left.pop()
    
    
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
R_MIN = 6             # minimum attack radius for IEEE300
R_MAZ = 8             # maximum attack radius for IEEE300
SENSOR_SIGMA = 0.01   # match the sigma used in the init_dataset
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
    
    _, _, T_x = load_tobeAttacked(dataset_config["G"], irrelevant_buses, 1, 
                                 entry_bus, radius, dataset_config["number_nodes"])
    
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
    def __init__(self, Ybus, baseMVA):
        self.Ybus = Ybus
        self.baseMVA = baseMVA
    

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
        
        # Mask to train only chosen to be attacked nodes:
        mask_training = torch.zeros_like(V.data)
        mask_training[T_x] = 1.0
        
        # Define optimizer and loss
        optimizer = torch.optim.Adam([V, th], lr=ETA)
        loss = None
        
        # Define masks for T_z and T_x
        attacked = torch.zeros(n, dtype=torch.bool, device=device)
        attacked[T_x] = True
        keep = ~attacked  # non-attacked buses

        den_z = torch.sqrt(keep.sum().float() * 4.0) + 1e-12

        # Minimize loss from L = LAMBDA_Z * Lz - LAMBDA_Z * Lx
        for epoch in range(EPOCHS):
            # The goal of this training is to find a new x̃ that:
            #   Maximally corrupts the measurements in Tₓ (the attacked nodes),
            #   While keeping the measurements outside Tₓ (i.e. all i∉Tₓ) as close as possible to the originals.
            
            """
            # Do I understand and utilize T_z and T_x correctly? I think T_z might not be what I think it is.
            # Is my measurement_model function correct conceptually?  
            
            # How can I make Lz also affect the loss?
            #   Maybe avoiding the use of pandapower and simply using the formula
            #   from the paper will help me calculate grds for za => Lz will have effect on the loss
            
            # Should I clamp into [0.9,1.1] and [−π,π] every epoch?
            
            # Is Lz and Lx calculated as follows: "Then, for each epoch, it obtains za using 
            #   h(x) and consequently calculates loss term Lz as a root mean
            #   squared error between za and zo, and Lx as a mean absolute
            #   error between ˇx and ˆx. Eventually"      
            
            """
            
            optimizer.zero_grad()
        
            # clamp into [0.9,1.1] and [−π,π]
            V.data.clamp_(0.9, 1.1)
            th.data.clamp_(-np.pi, np.pi)
        
            # build phasors and analytic measurements
            x_complex = V * torch.exp(1j * th)
            za = measurement_model(x_complex, self.Ybus, self.baseMVA)
        
            # compute per-paper losses
            z0_mat = z0_t.view(n, 4)   # reshape honest z

            Lz = torch.linalg.vector_norm((za[keep] - z0_mat[keep]).reshape(-1)) / den_z
            
            Lx = torch.mean(torch.abs(x_complex[attacked] - x_hat_t[attacked]))
            
            L = LAMBDA_Z * Lz - LAMBDA_X * Lx
            L.backward()
            
            # zero out grads in the non-attacked nodes
            V.grad  *= attacked.float()
            th.grad *= attacked.float()
            
            # Update parameters based on the gradients (only nodes from T_x == 1 have influence)
            optimizer.step()
            
            loss = L.item()
        
            if loss < TAU_LOSS:
                break
        
        # Compute complex state variable
        x_c_opt = V * torch.exp(1j * th)
        #x_c_opt = (V * torch.exp(1j*th)).detach().cpu()
        
        # Get new Z
        za_opt = measurement_model(x_c_opt, self.Ybus, self.baseMVA)
        
        return za_opt.detach().cpu().numpy(), loss

# Generate function per timestep
def step_Ao( attacker, z0, x_hat, net, t):    
    # Initialize V and th
    V = abs(x_hat[:, 2])
    th = x_hat[:, 3] * np.pi/180
    
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
    y[T_x] = 1 # TODO: ! make sure the target is correct
    
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
    # Generate Ybus and base unit of measurements from the current network
    net = pn.case300()
    pp.runpp(net)                              # just to build net._ppc
    
    Y_ppc = net._ppc['internal']['Ybus'].toarray()
    pd2ppc = net._pd2ppc_lookups['bus']        # map: pandapower index -> ppc index
    Y_aligned = Y_ppc[pd2ppc][:, pd2ppc]       # reindex Ybus into pandapower order
    
    Ybus = torch.tensor(Y_aligned, dtype=torch.complex64, device=device)
    baseMVA = net.sn_mva

    # Initialize the attacker with the Ybus and the unit computed
    attacker = Attacker_Ao(Ybus, baseMVA)

    # Input holders
    z0 = np.load(f"../init_dataset/Z_{t_load}.npy")
    x_hat = np.load(f"../init_dataset/X_{t_load}.npy")
    
    # Flatten z0 for future calculations
    z0 = z0.flatten()
    
    # Compute Z and X
    z_att, x_est, y = step_Ao(attacker, z0, x_hat, net, t_save)
    
    # Exclude Voltage magnitude and angle
    z_att = z_att[:, :2]
    
    # Add noise to match the other FDIAs 
    # (init_dataset adds noise to Z and other FDIAs reuse Z but Ao doesn't) 
    z_att += np.random.normal(0.0, np.abs(z_att)*SENSOR_SIGMA, size=z_att.shape)
    
    
    # Save current attacked sample
    np.save(f'../final_dataset/Z_{t_save}.npy', z_att)
        
    # Estimeated X
    #np.save(f'../final_dataset/X_{t_save}.npy', x_est)
    
        # Attach "attacked" graph-level flag to the target
    y = np.concatenate([y, [1]])
    np.save(f'../final_dataset/target_{t_save}.npy', y)
    
    return t_save  # for tracking progress


def generate_Ao(start_id, finish_id, max_workers=2):
    """
    # Un-paralleled generation for debugging
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
    # Get random timestep, entry bus, and radius of the attack
    t_load = returnTimeStep()
    p = np.random.randint(0, dataset_config["number_nodes"])
    r = np.random.randint(R_MIN, R_MAZ+1)
    
    # Load a node features and a mask with the buses to be attacked marked as True
    Z, mask, _ = load_tobeAttacked(dataset_config["G"], irrelevant_buses, t_load, p, r, 
                                dataset_config["number_nodes"])
  
    # - Ad data subset -
    print("Ad attack chosen")
        
    # Extract all node features
    P = Z[:, 0].copy()
    Q = Z[:, 1].copy()
    
    
    # Don't follow outliners, so that the attack is not as easy to detect
    # Trim outliers (5–95%)
    P_lo, P_hi = np.percentile(P, [5, 95])
    Q_lo, Q_hi = np.percentile(Q, [5, 95])
    
    P_trim = P[(P >= P_lo) & (P <= P_hi)]
    Q_trim = Q[(Q >= Q_lo) & (Q <= Q_hi)]
    
    P_mu, P_std = P_trim.mean(), P_trim.std(ddof=0)
    Q_mu, Q_std = Q_trim.mean(), Q_trim.std(ddof=0)
    
    # Avoid zero/NaN std
    eps = 1e-6
    P_std = float(max(P_std, eps))
    Q_std = float(max(Q_std, eps))
    
    k = int(mask.sum())
    new_P = np.random.normal(P_mu, P_std, size=k)
    new_Q = np.random.normal(Q_mu, Q_std, size=k)
    
    # Keep within trimmed bounds
    new_P = np.clip(new_P, P_lo, P_hi)
    new_Q = np.clip(new_Q, Q_lo, Q_hi)
    
    # Apply to attacked buses
    idx = np.where(mask)[0]
    Z[idx, 0] = new_P
    Z[idx, 1] = new_Q
        
    """
    # Apply the distribution-based attack on the marked buses
    Z[mask, 0] = np.random.normal(P.mean(), P.std(), size=mask.sum())
    Z[mask, 1] = np.random.normal(Q.mean(), Q.std(), size=mask.sum())
    """

    # Exclude Voltage magnitude and angle
    Z = Z[:, :2]
  
    # Generate target(expected output) for multi-label supervised learning
    target = np.concatenate([mask.astype(int), [1]])
  
    # Save the modified files
    saveNetwork(Z, target, t_save)


def generate_As(t_save):
    # Get random timestep, entry bus, and radius of the attack
    t_load = returnTimeStep()
    p = np.random.randint(0, dataset_config["number_nodes"])
    r = np.random.randint(R_MIN, R_MAZ+1)
    
    # Load a node features and a mask with the buses to be attacked marked as True
    Z, mask, _ = load_tobeAttacked(dataset_config["G"], irrelevant_buses, t_load, p, r, 
                                dataset_config["number_nodes"])
  
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


def generate_Ar(tau, t_save, t_load):
    # Get random entry bus, and radius of the attack
    p = np.random.randint(0, dataset_config["number_nodes"])
    r = np.random.randint(R_MIN, R_MAZ+1)
    
    # Load a node features and a mask with the buses to be attacked marked as True
    Z, mask, _ = load_tobeAttacked(dataset_config["G"], irrelevant_buses, t_load, p, r, 
                                dataset_config["number_nodes"])
    
    # Load data TAU steps before
    Z_tau, _, _ = load_tobeAttacked(dataset_config["G"], irrelevant_buses, 
                                t_load-tau, p, r, dataset_config["number_nodes"])
  
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
    # Get a random timestep
    t_load = returnTimeStep()
    
    # Load a node features and a mask filled with False values
    Z, mask = load_honest(t_load, dataset_config["number_nodes"])
    
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
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    
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
    prepared_Ar = list()
    tobe_Returned = list()
    tau = 4
    
    # Gather IDs to be used
    while(len(prepared_Ar) != 720):
        t_load = returnTimeStep()
        if(t_load-tau < 1 ):
            tobe_Returned.append(t_load)
        else:
            prepared_Ar.append(t_load)
    
    # Push bask unused IDs
    while(len(tobe_Returned) != 0):
        timesteps_left.append(tobe_Returned.pop())
        
    # Generate the samples
    for t in range(30961, 31681):
      print(f"\nCurrent cycle: #{t}")
      generate_Ar(tau, t, prepared_Ar.pop())
    
    # -- Generate honest samples (not-attacked) --
    for t in range(31681, 34561):
      print(f"\nNot-attacked, Current cycle: #{t}")
      generate_honest(t)
  
    if( dataset_config["standardize_data"] ):
        # ! ---- Standardization ---- !
        # Load in the training dataset
        root = "../final_dataset"
        Z_list = []
        
        for i in range(1, 23041):
            file_path = os.path.join(root, f"Z_{i}.npy")
            if os.path.exists(file_path):  # check if file exists
                Z_list.append(file_path)
            else:
                raise RuntimeError(f"Z_{i}.npy doesn't exist ->  dataset is corrupted")
    
        # Compute mean & std
        sumv = np.zeros(2); sumq = np.zeros(2); count = 0
        for fn in Z_list:
            A = np.load(fn)            # shape (n,2) = (P,Q)
            sumv  += A.sum(axis=0)
            sumq += (A*A).sum(axis=0)
            count += A.shape[0]
        
        mean = sumv / count
        var  = sumq / count - mean**2
        std  = np.sqrt(var)
    
        print(f"Training dataset, mean = {mean:.5f},  std = {std:.5f}")
        
        # Load in all files:
        pattern = os.path.join(root, "Z_*.npy")
        files = glob.glob(pattern)
    
        # Standardize each file
        for fn in files:
            arr = np.load(fn)
            arr = (arr - mean) / std
            base = os.path.basename(fn)
            np.save(os.path.join(root, base), arr)
