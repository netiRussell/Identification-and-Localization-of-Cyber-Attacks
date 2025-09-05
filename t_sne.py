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
from pathlib import Path
import re

# TODO: to be deleted
import sys

# Local function to parse ids to be considered for the T-SNE
def parse_ids(spec: str):
    out = []
    for part in spec.replace(' ', '').split(','):
        if not part: 
            continue
        if '-' in part:
            a, b = part.split('-')
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))

# -- Global settings -- 
PERPLEXITY = 50
TITLE = "All"
ID_SPEC = "1-34560"
ID_SPEC_train = "1-23040"
ID_SPEC_train_Ao = "1-5761, 11512-23041"
ID_SPEC_train_Ad = "5761-23041"
ID_SPEC_test_ArAs = "30241-34561"

ID_SPEC_Ao = "1-5760,23041-24480,28801-29520"
ID_SPEC_Ad = "5761-11520,24481-25920,29521-30240"
ID_SPEC_As = "30241-30960"
ID_SPEC_Ar = "30961-31680"

# -- Define the config --
config = {
        "dataset_root": "./final_dataset",
    }

# -- Get all Y and X
root = Path(config["dataset_root"])
extract = lambda p, pref: int(re.search(rf"{pref}_(\d+)\.npy$", p.name).group(1))

Z_map = {extract(p, "X"): p for p in root.glob("X_*.npy")}
T_map = {extract(p, "target"): p for p in root.glob("target_*.npy")}

ids = [i for i in parse_ids(ID_SPEC) if i in Z_map and i in T_map]
all_X = [str(Z_map[i]) for i in ids]
all_Y = [str(T_map[i]) for i in ids]


# -- Prepare the dataset for Y --
# Define the TSNE
tsneY = TSNE(
    n_components=2,
    perplexity=PERPLEXITY,
    learning_rate=200,
    early_exaggeration=5,
    n_iter=1500,
    init='pca',
)

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
labels = (Y.sum(axis=1) > 0)               # bools
colors = ['green' if lab else 'black' for lab in labels]

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
plt.title(f"Y projection ({TITLE})")
plt.xlabel("t-SNE dim 1")
plt.ylabel("t-SNE dim 2")
plt.grid(False)
plt.show()


# -- Prepare the dataset for X --
# Define the TSNE
tsneX = TSNE(
    n_components=2,
    perplexity=PERPLEXITY,
    learning_rate=200,
    early_exaggeration=5,
    n_iter=1500,
    init='pca',
)


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
plt.title(f"X projection ({TITLE})")
plt.xlabel("t-SNE dim 1")
plt.ylabel("t-SNE dim 2")
plt.grid(False)
plt.show()
