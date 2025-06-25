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

# Implemented functions
from functions import saveNetwork, loadDataset, selectDevice

# TODO: to be deleteds
import sys


"""
- Do I implement the Lz and Lx formulas correctly?
- Do I need to change active, reactive powers as well in the measurement_model?


TODOs:
    - Multiple random restarts of attack() if loss > TAU_LOSS
    - Lz might not affect the loss because measurement_model is is not differentiable
    - Make sure that 0-inj and generator buses are disregarded while T_x representing a single "island"
    
    - Use Z for all the final dataset samples
    - Parallel generation
    - Generate dataset
    - Adapt dataset loading function
    - Start a new training
"""


dataset_config = {
    "normal_scaling": False,
    "number_nodes": 300
    }

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
    
    # Find generator buses: 
    #    include all synchronous gens, static gens, and the external grid (slack)
    gen_buses = set()
    if hasattr(net, "gen"):
        gen_buses |= set(net.gen.bus.values)
    if hasattr(net, "sgen"):
        gen_buses |= set(net.sgen.bus.values)
    if hasattr(net, "ext_grid"):
        gen_buses |= set(net.ext_grid.bus.values)
    
    # Find zero-injection buses:
    #    any bus that has NO loads, NO gens, NO shunts, NO ext_grid
    #    (i.e. no “injection” elements attached)
    inj_buses = set()
    if hasattr(net, "load"):
        inj_buses |= set(net.load.bus.values)
    if hasattr(net, "shunt"):
        inj_buses |= set(net.shunt.bus.values)
    # include your generator buses too:
    inj_buses |= gen_buses
    
    # Filter out generator and zero-injection buses
    attacked_buses = [b for b in attacked_buses 
                      if b not in gen_buses and b not in inj_buses]
    
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
    
    print(loss)
    if loss > TAU_LOSS:
        # TODO: implement attack rerun
        print(f"[WARNING]: Sample #{t} didn't go below TAU_LOSS by {loss - TAU_LOSS}")
    
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

def generate_Ao(start_id, finish_id):
    # IEEE300 network
    net = pn.case300()
    attacker = Attacker_Ao(net)

    for t in range(start_id, finish_id):
        print(f"Current cycle: #{t}")
        print("Ao attack chosen")
        
        # Input holders
        z0 = np.load(f"../init_dataset/Z_{t}.npy")
        x_hat = np.load(f"../init_dataset/Z_{t}.npy")
        
        # Flatten z0 for future calculations
        z0 = z0.flatten()
        
        z_att, x_est, y = step_Ao(attacker, z0, x_hat, net, t)
        
        # Save current attacked sample
        np.save(f'../final_dataset/Z_{t}.npy', z_att)
        #np.save(f'../final_dataset/X_{t}.npy', x_est)
        np.save(f'../final_dataset/target_{t}.npy', y)


def generate_Ad(t):
    # Load a node features and a mask with the buses to be attacked marked as True
    Z, mask = loadDataset(t, True, dataset_config["number_nodes"])
  
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
    
    if( dataset_config["normal_scaling"] ):
        # Apply normal scaling [0,1]
        Z[:,0] = (Z[:,0] - P_min) / (P_max - P_min + 1e-8)
        Z[:,1] = (Z[:,1] - Q_min) / (Q_max - Q_min + 1e-8)
  
    # Generate target(expected output) for multi-label supervised learning
    target = np.concatenate([mask.astype(int), [1]])
  
    # Save the modified files
    saveNetwork(Z, target, t)


def generate_As(t):
    # Load a node features and a mask with the buses to be attacked marked as True
    Z, mask = loadDataset(t, True, dataset_config["number_nodes"])
  
    # - As data subset -
    print("As attack chosen")
      
    # Apply the scale-based attack on the marked buses
    Z[mask, 0] = Z[mask, 0] * np.random.uniform(0.8, 1.2, size=mask.sum())
    Z[mask, 1] = Z[mask, 1] * np.random.uniform(0.8, 1.2, size=mask.sum())
    
    # Exclude Voltage magnitude and angle
    Z = Z[:, :2]
    
    if( dataset_config["normal_scaling"] ):
        # Apply normal scaling [0,1]
        Z[:,0] = (Z[:,0] - P_min) / (P_max - P_min + 1e-8)
        Z[:,1] = (Z[:,1] - Q_min) / (Q_max - Q_min + 1e-8)
  
    # Generate target(expected output) for multi-label supervised learning
    target = np.concatenate([mask.astype(int), [1]])
  
    # Save the modified files
    saveNetwork(Z, target, t)


def generate_Ar(t):
    # Load a node features and a mask with the buses to be attacked marked as True
    Z, mask = loadDataset(t, True, dataset_config["number_nodes"])
    
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
    
    if( dataset_config["normal_scaling"] ):
        # Apply normal scaling [0,1]
        Z[:,0] = (Z[:,0] - P_min) / (P_max - P_min + 1e-8)
        Z[:,1] = (Z[:,1] - Q_min) / (Q_max - Q_min + 1e-8)
  
    # Generate target(expected output) for multi-label supervised learning
    target = np.concatenate([mask.astype(int), [1]])
  
    # Save the modified files
    saveNetwork(Z, target, t)


def generate_honest(t):
    # Load a node features and a mask filled with False values
    Z, mask = loadDataset(t, False, dataset_config["number_nodes"])
    
    # Exclude Voltage magnitude and angle
    Z = Z[:, :2]
    
    if( dataset_config["normal_scaling"] ):
        # Apply normal scaling [0,1]
        Z[:,0] = (Z[:,0] - P_min) / (P_max - P_min + 1e-8)
        Z[:,1] = (Z[:,1] - Q_min) / (Q_max - Q_min + 1e-8)
  
    # Generate target(expected output) for multi-label supervised learning
    target = np.concatenate([mask.astype(int), [0]])
  
    # Save the modified files
    saveNetwork(Z, target, t)




# # # # # # # # # # # #
# ---- Main code ---- #
# # # # # # # # # # # #


"""
# --- Compare Ad vs As influence ---
Z0, mask = loadDataset(100, True, dataset_config["number_nodes"])
Zd = np.copy(Z0)
    
# Extract all node features
P = Zd[:, 0]
Q = Zd[:, 1]

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
Zd[mask_P, 0] = np.random.normal(P_mu, P_std, size=mask_P.sum())
Zd[mask_Q, 1] = np.random.normal(Q_mu, Q_std, size=mask_Q.sum())

print(f"Zd: {np.mean(np.abs(Z0-Zd))}")


Zs = np.copy(Z0)
Zs[mask, 0] = Zs[mask, 0] * np.random.uniform(0.8, 1.2, size=mask.sum())
Zs[mask, 1] = Zs[mask, 1] * np.random.uniform(0.8, 1.2, size=mask.sum())
print(f"Zs: {np.mean(np.abs(Z0-Zs))}")

sys.exit()
"""

if __name__ == '__main__':
    
    # !---- Get max and min for normal scaling ----!
    max_num_timesteps = 44634
    
    if( dataset_config["normal_scaling"] ):
        # -- Prepare normal scaling --
        # Define initial mins and maxs
        Z0 = np.load(f"../init_dataset/Z_{0}.npy", mmap_mode="r")
        P_min = Z0[:, 0].min()
        P_max = Z0[:, 0].max()
        Q_min = Z0[:, 1].min()
        Q_max = Z0[:, 1].max()
        
        # Get the final mins and maxs
        for t in range(1, max_num_timesteps):
            print(f"Current graph: {t}")
            
            # Load a graph
            Z = np.load(f"../init_dataset/Z_{t}.npy")
            
            # Extract the node features to be attacked
            p = Z[:, 0]
            q = Z[:, 1]
        
            # Update global mins/maxs
            cur_p_min, cur_p_max = p.min(), p.max()
            cur_q_min, cur_q_max = q.min(), q.max()
        
            if cur_p_min < P_min:
                P_min = cur_p_min
            if cur_p_max > P_max:
                P_max = cur_p_max
            if cur_q_min < Q_min:
                Q_min = cur_q_min
            if cur_q_max > Q_max:
                Q_max = cur_q_max
    
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
    
