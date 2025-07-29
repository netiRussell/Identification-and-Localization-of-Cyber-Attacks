# Limit to two threads max
import os
os.environ["OMP_NUM_THREADS"]   = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"]   = "4"

from sklearn.manifold import TSNE
from dataset import FDIADataset
from dataset_generators.functions import preparePyGDataset
import matplotlib.patches as mpatches
import numpy as np
import glob
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits

# TODO: to be deleted
import sys



# -- Define the config --
config = {
        "dataset_root": "./final_dataset_NotStandard",
    }

# -- Prepare the dataset for Y --
# Define the TSNE
tsneY = TSNE(
    n_components=2,
    perplexity=25,
    learning_rate=200,
    early_exaggeration=5,
    n_iter=1500,
    init='pca',
)

# Get all the Y samples
all_Y = sorted(glob.glob(config["dataset_root"] + "/target_*.npy"))
# Prepare the Y holder
Y = np.zeros((len(all_Y), 301), dtype="float")

# Load all the Y samples to the holder
for index, sample in enumerate(all_Y):
    print(f"Current sample: {index}")
    Y[index] = np.load(sample)

# -- Generate Y T-SNE --
print("Preparing Y T-SNE...")
with threadpool_limits(limits=4):
    Y_embedded = tsneY.fit_transform(Y)
print(f"Y TSNE shape: {Y_embedded.shape}")

# -- Get an array of corresponding colors --
labels = (Y.sum(axis=1) > 0).astype(int)
colors = ['green' if (sample.sum() > 0).astype(int) else 'black' for sample in labels]

# -- Visualize Y --
plt.figure(figsize=(8, 8))
plt.scatter(Y_embedded[:, 0], Y_embedded[:, 1],
            c=colors,  # Array that specifiec attacked and healthy nodes
            s=5,       # marker size
            alpha=0.7  # transparency
           )
plt.legend(handles=[
        mpatches.Patch(color='black', label='No Attack'),
        mpatches.Patch(color='green',      label='Attack')
    ])
plt.title("Y projection")
plt.xlabel("t-SNE dim 1")
plt.ylabel("t-SNE dim 2")
plt.grid(False)
plt.show()


# -- Prepare the dataset for X --
# Define the TSNE
tsneX = TSNE(
    n_components=2,
    perplexity=25,
    learning_rate=200,
    early_exaggeration=5,
    n_iter=1500,
    init='pca',
)

# Get all the X samples
all_X = sorted(glob.glob(config["dataset_root"] + "/Z_*.npy"))
# Prepare the X holder
X = np.zeros((len(all_X), 300, 2), dtype="float")

# Load all the X samples to the holder
for index, sample in enumerate(all_X):
    print(f"Current sample: {index}")
    X[index] = np.load(sample)

# Reshape the X holder to fit the TSNE function
X = X.reshape(X.shape[0], 300*2)

# -- Generate X T-SNE --
print("Preparing X T-SNE...")
with threadpool_limits(limits=4):
    X_embedded = tsneX.fit_transform(X)
print(f"X TSNE shape: {X_embedded.shape}")

# -- Visualize X -- 
plt.figure(figsize=(8, 8))
plt.scatter(X_embedded[:, 0], X_embedded[:, 1],
            c=colors,  # Array that specifiec attacked and healthy nodes
            s=5,       # marker size
            alpha=0.7  # transparency
           )
plt.legend(handles=[
        mpatches.Patch(color='black', label='No Attack'),
        mpatches.Patch(color='green',      label='Attack')
    ])
plt.title("X projection")
plt.xlabel("t-SNE dim 1")
plt.ylabel("t-SNE dim 2")
plt.grid(False)
plt.show()
