import pandapower as pp
import pandapower.networks as ppn
import pandapower.topology as pptop
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

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

# -- Create a network with 2848 busses --
net = ppn.case2848rte()

# -- Scale load and generation -- 
for tbl in ["load", "sgen", "gen"]:
    df = getattr(net, tbl)
    if len(df) == 0:
        continue
    
    # Generate an array of scale factors for all the elements
    sf = np.random.uniform(0.8, 1.2, size=len(df))

    # Scale active power
    df["p_mw"] *= sf

    # If reactive power column exists, scale it as well
    if "q_mvar" in df.columns:
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

measurements = net.res_line[["p_from_mw", "q_from_mvar", "p_to_mw", "q_to_mvar"]]
noise = np.random.normal(loc=0.0, scale=0.01, size=measurements.shape)

net.res_line[["p_from_mw", "q_from_mvar", "p_to_mw", "q_to_mvar"]] += measurements * noise

# -- Turn the net into a graph --
G = pptop.create_nxgraph(
    net,
    respect_switches=False,
    include_impedances=True,
    calc_branch_impedances=True,
    branch_impedance_unit="ohm"
)

# Annotate each bus
for b in G.nodes():
    r = net.res_bus.loc[b]
    G.nodes[b].update({
        "p_mw":    r["p_mw"],
        "q_mvar":  r["q_mvar"],
        "vm_pu":   r["vm_pu"],
        "va_degree": r["va_degree"]
    })

# Visualize
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_size=2848, edge_color='gray')
plt.title("Pandapower Network as Graph")
plt.show()

# -- Save the graph --



# TODO: Finish a single cycle
# TODO: turn it into multiple cycles

'''
Questions:
1) Do I add noise to the correct columns?
1) Do I scale the correct columns?
'''