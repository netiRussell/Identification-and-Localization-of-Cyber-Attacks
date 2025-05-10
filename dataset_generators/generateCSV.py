# ! Use this file to generate easy to open csv files for analysis of data structure

import pandas as pd
import numpy as np

outputName = "target_attacked"
filePath = "../Ad_dataset/target4.npy"
file = np.load(filePath)

"""
# Print out the content of the file without CSV generation
import sys

print(file)
sys.exit()
"""

# Convert array into dataframe and
# save the dataframe as a csv file 
DF = pd.DataFrame(file) 
DF.to_csv(f"./{outputName}.csv")