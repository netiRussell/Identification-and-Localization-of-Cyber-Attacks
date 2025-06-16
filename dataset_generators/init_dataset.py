import pandapower as pp
import pandapower.networks as ppn
import pandapower.topology as pptop
import networkx as nx
import numpy as np
import copy

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

    """
    # -- Visualize -- 
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=2848, edge_color='gray')
    plt.title("Pandapower Network as Graph")
    plt.show()
    """
    
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
    """
    import pandas as pd
    # Convert array into dataframe and
    # save the dataframe as a csv file 
    DF = pd.DataFrame(X) 
    DF.to_csv(f"../init_dataset/X{i}.csv")
    """

    # Save the feature nodes of the current instance
    np.save(f"../init_dataset/x{i}", X)
    
    
