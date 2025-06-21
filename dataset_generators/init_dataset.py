import pandapower as pp
import pandapower.networks as pn
from pandapower.estimation import estimate
from pandapower.create import create_measurement
import pandapower.topology as pptop
import networkx as nx
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import copy

# Should I scale both react and act power with the same scaling factor?
# How to get IEEE bus systems?
# Is it ok to use absolute value for the std when conducting 1% additive noise: scale=abs(z * sigma_n)
# Is my PSSE function correct, is flat(sets 1.0 p.u. / 0° for all buses) a good values for init?
    # Is it ok to use WLS as the algorithm for PSSE?
    # Is it ok to use std=abs(p)*sigma_n when creating measurements?
# For some samples, AC powerflow alg doesn't converge, is that normal?
# Are you sure X is used as node features? In the papers, attack targets Z0

"""
Each transmission line or branch connects two buses:
    A “from” bus (start node)
    A “to” bus (end node)

Bus - any point where components connect electrically.

Active load - energy converted into useful work like motion, light, or heat. 

Reactive load - energy stored and released by inductors or capacitors
Doesn't do useful work, but is essential for voltage stability.

Together, these loads determine the apparent power (S)
"""

# TODO: delete after debugging is fully complete.
import sys

def scalerGenerator():
    # -- Scaler generation --
    import pandas as pd
    import glob
    from sklearn.preprocessing import StandardScaler  # or MinMaxScaler
    
    # Load & concat
    files = glob.glob("./csv/*.csv")
    dfs = []
    for fn in files:
        # assume each CSV has columns ["Time Stamp","Load"]
        df = pd.read_csv(fn, parse_dates=["Time Stamp"], index_col="Time Stamp")
        dfs.append(df)
    
    full = pd.concat(dfs).sort_index()
    full = full[~full.index.duplicated()]
    
    # Resample to 1-min and interpolate
    #    this will create every minute from start to end
    full_1min = full.resample("1T").interpolate(method="time")
    
    # 3. Fit a scaler
    loads = full_1min["Load"].values.reshape(-1, 1)
    scaler = StandardScaler().fit(loads)
    
    # Normalize the loads
    norm_vec = scaler.transform(loads).ravel()
    
    # Save the final vector
    np.save("nyiso_normalized_load.npy", norm_vec)

# -- Initial Configurations --
# IEEE bus systems
N = {"14": pn.case14(),
     "118": pn.case118(),
     "300": pn.case300()}
# Number of timesteps
T = 36000 
# Scaling coefficients
k = 0.1
sigma_s=0.03
# Noise coefficient
sigma_n = 0.01
# Scaling vector
S = np.load("S.npy")

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
    # If it doesn't converge, skip this timestep
    try:
        estimate(
            net,
            algorithm="wls",
            init="results",
            tolerance=1e-6,
            maximum_iterations=20,
            calculate_voltage_angles=True
        )
    except Exception as e:
        #print(f"The estimation didn't converge! {e}")
        return None

    # Build estimated feature matrix
    res = net.res_bus_est.sort_index()
    x = np.vstack([
        res.loc[bus, ["p_mw", "q_mvar", "vm_pu", "va_degree"]].values
        for bus in nodelist
    ])
    return x

# Function for a single time-step generation
def generate(net, S_t, G, nodelist, base_p, base_q):
    # Generate the scale factor
    sf = np.random.normal(loc=(1 + k * S_t), scale=sigma_s, size=len(net.load.index))
    # Bound sf within 0.7 and 1.3
    sf = np.where(sf > 1.3, 1.3, sf)
    sf = np.where(sf < 0.7, 0.7, sf)
    

    # Scale the buses
    net.load["p_mw"] = base_p * sf
    net.load["q_mvar"] = base_q * sf
    
    # Run the AC powerflow algorithm
    # If it doesn't converge, skip this timestep
    try:
        pp.runpp(net, init="results")
    except Exception as e:
        #print(f"The AC pf didn't converge! {e}")
        return None, None

    # Copy bus‐measurements into node attributes
    for b in G.nodes():
        r = net.res_bus.loc[b]
        G.nodes[b].update({
            "p_mw":    r["p_mw"],
            "q_mvar":  r["q_mvar"],
            "vm_pu":   r["vm_pu"],
            "va_degree": r["va_degree"]
        })
    
    # Build the measurement matrix; shape: [N, 4]
    attrs = ["p_mw","q_mvar","vm_pu","va_degree"]
    z = np.vstack([
        [ G.nodes[n][k] for k in attrs ]
        for n in nodelist
    ])

    # Add 1% relative uniform noise 
    z = np.random.normal(loc=z, scale=abs(z * sigma_n))
    
    # Get estimated state
    x = run_psse(net, z, sigma_n, nodelist)

    return z, x



def worker(t):
    # each worker builds its own IEEE-300 net
    net  = pn.case300()
    pp.runpp(net)
    G    = pptop.create_nxgraph(net,
                respect_switches=False,
                include_impedances=True,
                calc_branch_impedances=True,
                branch_impedance_unit="ohm")
    nodelist = sorted(G.nodes())
    base_p   = net.load["p_mw"].values.copy()
    base_q   = net.load["q_mvar"].values.copy()

    z, x = generate(net, S[t], G, nodelist, base_p, base_q)
    if z is None or x is None:
        return None
    return z, x


# # # # # # # # # # # #
# ---- Main code ---- #
# # # # # # # # # # # #
# -- Select network --
net = N["300"]
base_p  = net.load["p_mw"].values.copy()
base_q  = net.load["q_mvar"].values.copy()

# -- Run the AC powerflow algorithm to generate the Y bus --
pp.runpp(net)

# -- Extract the Ybus graph representation --
Ybus = net._ppc['internal']['Ybus'].toarray()   # complex-valued
# Reorder rows/cols to match pandapower’s bus indexing
pd2ppc = net._pd2ppc_lookups['bus']
Ybus = Ybus[pd2ppc][:, pd2ppc]

W = np.abs(Ybus) # Get real values
np.fill_diagonal(W, 0) # Exclude node-wise self-looping

# Extract sparse edges
rows, cols = np.nonzero(W)
edge_indices = np.vstack([rows, cols])          # shape [2, E]
weights = W[rows, cols]                   # shape [E]

# -- Save the net structure as edge indices and correpsonding weights  --
np.save("../init_dataset/edge_indices", edge_indices)
np.save("../dataset/edge_indices", edge_indices)

np.save("../init_dataset/weights", weights)
np.save("../dataset/weights", weights)

# -- Turn the net into a graph --
G = pptop.create_nxgraph(
    net,
    respect_switches=False,
    include_impedances=True,
    calc_branch_impedances=True,
    branch_impedance_unit="ohm"
)

# -- Sort the nodes to make sure node-order is consistent --
nodelist = sorted(G.nodes())

# -- Extract the weighted adjacency matrix -- 
W_mat = nx.to_numpy_array(G, nodelist=nodelist, weight="z_ohm")

# -- Collect direct neighbors of each node for the future use of BFS --
neighbors = [
        list(np.nonzero(W_mat[i] != 0)[0])
        for i in range(W_mat.shape[0])
    ]

# -- Save direct neighbors of each node --
# (Saved as an object because of inconsistent neighbors[:, 1].shape)
np.save("../init_dataset/neighbors", np.array(neighbors, dtype=object))

# -- For L timesteps, generate a sample --
if __name__ == "__main__":
    idx = 1
    with ProcessPoolExecutor(max_workers=6) as exe:
        # map ensures all 36 000 t’s are processed in parallel
        for result in exe.map(worker, range(len(S))):
            if result is None:
                continue
            z, x = result
            print(f"Generated sample {idx}")
            np.save(f"../init_dataset/Z_{idx}.npy", z)
            np.save(f"../init_dataset/X_{idx}.npy", x)
            idx += 1
    
    print(f"Done. Saved {idx-1} samples.")



"""
def singleCycle( net, nodelist, load_scaler, gen_scaler, i ):
    # -- Scale the load -- 
    # Scale active power
    net.load["p_mw"] *= load_scaler[i]

    # Scale reactive power
    net.load["q_mvar"] *= load_scaler[i]

    # -- Run AC powerflow algorithm --
    pp.runpp(net)

    # -- Build the measurement matrices --
    meas_line = net.res_line[ ["p_from_mw","q_from_mvar","p_to_mw","q_to_mvar"] ].values
    meas_bus  = net.res_bus[ ["p_mw","q_mvar","vm_pu","va_degree"] ].values

    # -- Add 1% relative uniform noise --
    noise_line = np.random.uniform(-0.01,0.01, size=meas_line.shape)
    noise_bus  = np.random.uniform(-0.01,0.01, size=meas_bus.shape)

    net.res_line[ ["p_from_mw","q_from_mvar","p_to_mw","q_to_mvar"] ] += meas_line * noise_line
    net.res_bus [ ["p_mw","q_mvar","vm_pu","va_degree"]  ] += meas_bus  * noise_bus

    # Turn the current net into a NetworkX graph with ohm‐weights
    G = pptop.create_nxgraph(net,
        respect_switches=False,
        include_impedances=True,
        calc_branch_impedances=True,
        branch_impedance_unit="ohm"
    )

    # Copy bus‐measurements into node attributes
    for b in G.nodes():
        r = net.res_bus.loc[b]
        G.nodes[b].update({
            "p_mw":    r["p_mw"],
            "q_mvar":  r["q_mvar"],
            "vm_pu":   r["vm_pu"],
            "va_degree": r["va_degree"]
        })
    
    # Extract X in bus‐index order
    attrs = ["p_mw","q_mvar","vm_pu","va_degree"]
    X = np.vstack([
        [ G.nodes[n][k] for k in attrs ]
        for n in nodelist
    ])
    return X




# # # # # # # # # # # #
# ---- Main code ---- #
# # # # # # # # # # # #

# -- Create a network with 2848 busses -- 
net = ppn.case2848rte()

# -- Run a power flow calculation on the initial setup of the network using the pandapower library -- 
pp.runpp(net)

# -- Extract the Ybus graph representation --
Ybus = net._ppc['internal']['Ybus'].toarray()   # complex-valued
# Reorder rows/cols to match pandapower’s bus indexing
pd2ppc = net._pd2ppc_lookups['bus']
Ybus = Ybus[pd2ppc][:, pd2ppc]

W = np.abs(Ybus) # Get real values
np.fill_diagonal(W, 0) # Exclude node-wise self-looping

# Extract sparse edges
rows, cols = np.nonzero(W)
edge_indices = np.vstack([rows, cols])          # shape [2, E]
weights = W[rows, cols]                   # shape [E]

# -- Save the net structure as edge indices and correpsonding weights  --
np.save("../init_dataset/edge_indices", edge_indices)
np.save("../dataset/edge_indices", edge_indices)

np.save("../init_dataset/weights", weights)
np.save("../dataset/weights", weights)

# -- Turn the net into a graph --
G = pptop.create_nxgraph(
    net,
    respect_switches=False,
    include_impedances=True,
    calc_branch_impedances=True,
    branch_impedance_unit="ohm"
)

# -- Sort the nodes to make sure node-order is consistent --
nodelist = sorted(G.nodes())

# -- Extract the weighted adjacency matrix -- 
W_mat = nx.to_numpy_array(G, nodelist=nodelist, weight="z_ohm")

# -- Collect direct neighbors of each node for the future use of BFS --
neighbors = [
        list(np.nonzero(W_mat[i] != 0)[0])
        for i in range(W_mat.shape[0])
    ]

# -- Save direct neighbors of each node --
# (Saved as an object because of inconsistent neighbors[:, 1].shape)
np.save("../init_dataset/neighbors", np.array(neighbors, dtype=object))

# -- Configure scalers --
L = 36000
n_load = len(net.load)
n_gen  = len(net.gen)
load_scaler = np.random.uniform(0.8, 1.2, size=(L, n_load))
gen_scaler  = np.random.uniform(0.8, 1.2, size=(L, n_gen))

# -- Run the AC algorithm with differently scaled loads L times --
for i in range(L):
    # -- Single cycle output --
    print(f"Current instance: #{i}")

    # Copy the initial network and
    # get the network data
    net_copy = copy.deepcopy( net )
    X = singleCycle( net_copy, nodelist, load_scaler, gen_scaler, i )


    # -- Save the data subsets --
    # Save the feature nodes of the current instance
    np.save(f"../init_dataset/x{i}", X)
"""
    
