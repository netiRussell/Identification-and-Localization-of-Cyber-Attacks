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

def singleCycle( net ):
    
    # -- Scale the load -- 
    df = net["load"]
    
    # Generate an array of scale factors for all the elements
    sf = np.random.uniform(0.8, 1.2, size=len(df))

    # Scale active power
    df["p_mw"] *= sf

    # Scale reactive power
    df["q_mvar"] *= sf


    # -- Run AC powerflow algorithm --
    pp.runpp(net)


    # -- Add 1% Guassian noise --
    """
    Uniform 1% option:
    noise = np.random.uniform(-0.01, 0.01, size=bus_meas.shape)

    p_from_mw - Real (active) power in megawatts (MW) flowing from the “from” bus of the line.
    q_from_mvar - Reactive power in megavolt-amperes reactive (MVAr) flowing from the “from” bus.
    p_to_mw - Real power in MW flowing into the “to” bus of the line.
    q_to_mvar - Reactive power in MVAr flowing into the “to” bus.
    """

    measurements_line = net.res_line[["p_from_mw", "q_from_mvar", "p_to_mw", "q_to_mvar"]]
    measurements_bus = net.res_bus[["vm_pu", "va_degree"]]
    noise_line = np.random.normal(loc=0.0, scale=0.01, size=measurements_line.shape)
    noise_bus = np.random.normal(loc=0.0, scale=0.01, size=measurements_bus.shape)

    net.res_line[["p_from_mw", "q_from_mvar", "p_to_mw", "q_to_mvar"]] += measurements_line * noise_line
    net.res_bus[["vm_pu", "va_degree"]] += measurements_bus * noise_bus




    # -- Turn the net into a graph --
    # Generate the graph
    G = pptop.create_nxgraph(
        net,
        respect_switches=False,
        include_impedances=True,
        calc_branch_impedances=True,
        branch_impedance_unit="ohm"
    )

    # Add the needed attributed to each bus
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

    # Extract the node values(P,Q,V,theta)
    attrs = ["p_mw", "q_mvar", "vm_pu", "va_degree"]
    X = np.array([
        [G.nodes[n][key] for key in attrs]
        for n in list(G.nodes())
    ])

    return W_mat, X




# # # # # # # # # # # #
# ---- Main code ---- #
# # # # # # # # # # # #

# -- Create a network with 2848 busses -- 
net = ppn.case2848rte()

# -- Turn in into a graph --
G = pptop.create_nxgraph(
    net,
    respect_switches=False,
    include_impedances=True,
    calc_branch_impedances=True,
    branch_impedance_unit="ohm"
)

# -- Extract the weighted adjacency matrix -- 
W_mat = nx.to_numpy_array(G, weight="z_ohm")

# -- Save the layout as a weighted adjacency matrix --
np.save(f"../init_dataset/w_mat", W_mat)
np.save(f"../Ad_dataset/w_mat", W_mat)
np.save(f"../As_dataset/w_mat", W_mat)

# -- Run the AC algorithm with differently scaled loads 36000 times --
for i in range(36000):
    # -- Single cycle output --
    print(f"Current instance: #{i}")

    # Copy the initial network and
    # get the network data
    net_copy = copy.deepcopy(net)
    W_mat, X = singleCycle( net_copy )


    # -- Save the data subsets --
    """
    import pandas
    # Convert array into dataframe and
    # save the dataframe as a csv file 
    DF = pd.DataFrame(X) 
    DF.to_csv(f"../init_dataset/X{i}.csv")
    """

    # Save the feature nodes of the current instance
    np.save(f"../init_dataset/x{i}", X)



'''
Questions:
...

'''