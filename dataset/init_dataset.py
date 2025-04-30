import pandapower as pp
import pandapower.networks as ppn
import pandapower.topology as pptop
import numpy as np

# Create a network with 2848 busses
net = ppn.case2848rte()

# Scale load and generation
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

# Run AC powerflow algorithm
pp.runpp(net)

# Extract the bus measurements
measurments = net.copy()


"""
# Add 1% Guassian noise
noise = np.random.normal(loc=0.0, scale=0.01, size=measurments.shape)
# noise = np.random.uniform(-0.01, 0.01, size=bus_meas.shape)
measurments += measurments.values * noise
"""




# TODO: Finish a single cycle
# TODO: turn it into multiple cycles

'''
Questions:
1) Do we save P and Q or V and the angle? If P and Q then from where?
2) Did I collect P and Q from the correct tables?

3) Noise is being added to what columns?
'''